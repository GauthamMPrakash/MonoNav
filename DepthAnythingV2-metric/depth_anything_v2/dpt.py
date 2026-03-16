import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from contextlib import nullcontext

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        max_depth=20.0
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.max_depth = max_depth
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        # Runtime knobs for high-throughput inference.
        self._transform_cache = {}
        self._autocast_enabled = False
        self._autocast_dtype = torch.float16
        self._channels_last_enabled = False
        self._model_in_half = False
        self._resize_interpolation = cv2.INTER_LINEAR
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        
        return depth.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size):
        image, (h, w) = self.image2tensor(raw_image, input_size)

        device_type = image.device.type
        amp_ctx = (
            torch.autocast(device_type=device_type, enabled=True, dtype=self._autocast_dtype)
            if self._autocast_enabled and device_type in ("cuda", "cpu")
            else nullcontext()
        )

        with amp_ctx:
            depth = self.forward(image)
        
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.float().cpu().numpy()

    def configure_runtime(
        self,
        use_autocast=None,
        autocast_dtype="float16",
        use_channels_last=True,
        use_half_precision_model=False,
        fast_resize=True,
    ):
        # Must be called after moving model to target device.
        device = next(self.parameters()).device

        if use_autocast is None:
            use_autocast = device.type == "cuda"

        autocast_dtype = str(autocast_dtype).lower()
        if autocast_dtype in ("bf16", "bfloat16"):
            self._autocast_dtype = torch.bfloat16
        else:
            self._autocast_dtype = torch.float16

        if device.type == "cuda":
            self._autocast_enabled = bool(use_autocast)
        elif device.type == "cpu":
            # CPU autocast is practical only with bfloat16.
            self._autocast_enabled = bool(use_autocast and self._autocast_dtype == torch.bfloat16)
        else:
            self._autocast_enabled = False

        self._resize_interpolation = cv2.INTER_LINEAR if fast_resize else cv2.INTER_CUBIC

        if device.type == "cuda" and use_channels_last:
            self.to(memory_format=torch.channels_last)
            self._channels_last_enabled = True
        else:
            self._channels_last_enabled = False

        # Optional full model FP16. Autocast alone is usually safer.
        if device.type == "cuda" and use_half_precision_model:
            self.half()
            self._model_in_half = True
        else:
            self._model_in_half = False

        return {
            "device": str(device),
            "autocast": self._autocast_enabled,
            "autocast_dtype": "bfloat16" if self._autocast_dtype == torch.bfloat16 else "float16",
            "channels_last": self._channels_last_enabled,
            "model_half": self._model_in_half,
            "fast_resize": fast_resize,
        }

    def _get_transform(self, input_size):
        key = int(input_size)
        if key not in self._transform_cache:
            self._transform_cache[key] = Compose([
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=self._resize_interpolation,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        return self._transform_cache[key]
    
    def image2tensor(self, raw_image, input_size=518):
        transform = self._get_transform(input_size)

        h, w = raw_image.shape[:2]
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        device = next(self.parameters()).device
        if self._model_in_half:
            image = image.half()
        if self._channels_last_enabled:
            image = image.contiguous(memory_format=torch.channels_last)
        image = image.to(device, non_blocking=True)
        
        return image, (h, w)
