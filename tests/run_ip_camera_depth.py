import cv2
import matplotlib
import numpy as np
import torch
import os
import sys

# Add DepthAnythingV2-metric to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
metric_path = os.path.join(repo_root, "DepthAnythingV2-metric")
if metric_path not in sys.path:
    sys.path.insert(0, metric_path)

from depth_anything_v2.dpt import DepthAnythingV2

# -----------------------------
# ensure project root on path so utils package is importable
# -----------------------------
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# -----------------------------
# CONFIG (read from project config.yml)
# -----------------------------
from utils.utils import load_config
cfg = load_config(os.path.join(repo_root, "config.yml"))
STREAM_URL = cfg.get("camera_ip")
INPUT_SIZE = cfg.get("INPUT_SIZE")
CHECKPOINT = "../"+cfg.get("DA2_CHECKPOINT")
ENCODER = CHECKPOINT[-8:-4]
MAX_DEPTH = cfg.get("DA2_MAX_DEPTH", cfg.get("VoxelBlockGrid", {}).get("depth_max", 20))
SAVE_NUMPY = cfg.get("save_numpy", False)
PRED_ONLY = cfg.get("pred_only", False)
GRAYSCALE = cfg.get("grayscale", False)
OUTDIR = "./esp32_depth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

print(" ESP32 HTTP stream opened")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(" Frame not received, retrying...")
        continue

    # Show RGB stream immediately, before depth estimation.
    # cv2.imshow("ESP32 RGB Stream", frame)
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
    
    print(depth.min(), depth.max(), flush=True)

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
