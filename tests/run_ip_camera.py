import cv2
import matplotlib
import numpy as np
import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2

# -----------------------------
# CONFIG
# -----------------------------
STREAM_URL = "http://10.42.0.29:81/stream"  # YOUR ESP32 HTTP MJPEG stream
INPUT_SIZE = 336
OUTDIR = "./esp32_depth"
ENCODER = 'vitb'  # must match your checkpoint
CHECKPOINT = "../checkpoints/depth_anything_v2_metric_hypersim_vitb.pth"
MAX_DEPTH = 20
SAVE_NUMPY = False
PRED_ONLY = False
GRAYSCALE = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTDIR, exist_ok=True)

cmap = matplotlib.colormaps.get_cmap('Spectral')

# -----------------------------
# MODEL CONFIG
# -----------------------------
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Initialize the DepthAnythingV2 model and load the checkpoint
depth_anything = DepthAnythingV2(**{**model_configs[ENCODER], 'max_depth': MAX_DEPTH})
depth_anything.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
depth_anything = depth_anything.to(DEVICE).eval()
model_device = next(depth_anything.parameters()).device

# -----------------------------
# OPEN HTTP STREAM
# -----------------------------
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("Cannot open ESP32 HTTP stream")

print("✅ ESP32 HTTP stream opened")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received, retrying...")
        continue

    # Show RGB stream immediately, before depth estimation.
    cv2.imshow("ESP32 RGB Stream", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # -----------------------------
    # RUN DEPTH ESTIMATION
    # -----------------------------
    depth = depth_anything.infer_image(frame, INPUT_SIZE)
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    if SAVE_NUMPY:
        output_path = os.path.join(OUTDIR, f'frame_{frame_count:05d}_raw_depth_meter.npy')
        np.save(output_path, depth)

    # normalize for visualization
    
    print(depth.min())

    if GRAYSCALE:
        depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
    else:
        depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # combine with RGB if PRED_ONLY is False
    if not PRED_ONLY:
        split_region = np.ones((frame.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([frame, split_region, depth_vis])
    else:
        combined_result = depth_vis

    # display depth (or RGB + depth)
    cv2.imshow("ESP32 Metric Depth", combined_result)
    frame_count += 1

    # exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
