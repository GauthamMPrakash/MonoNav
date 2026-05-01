"""
  __  __                   _   _             
 |  \/  | ___  _ __   ___ | \ | | __ ___   __
 | |\/| |/ _ \| '_ \ / _ \|  \| |/ _` \ \ / /
 | |  | | (_) | | | | (_) | |\  | (_| |\ V / 
 |_|  |_|\___/|_| |_|\___/|_| \_|\__,_| \_/  
Copyright (c) 2023 Nate Simon
License: MIT
Authors: Nate Simon and Anirudha Majumdar, Princeton University
Project Page: https://natesimon.github.io/mononav

This script reads in a stream of RGB images, transforms them to the Kinect intrinsics,
and estimates metric depth using ZoeDepth.

The following are saved to file:
│   ├── <kinect_rgb_images> # images transformed to match kinect intrinsics
│   ├── <kinect_depth_images> # estimated depth (.npy for fusion and .jpg for visualization)

"""

import time
import os
import sys
import torch
import cv2

# Add ZoeDepth to path
# sys.path.insert(0, "ZoeDepth")
# from zoedepth.models.builder import build_model
# from zoedepth.utils.config import get_config

import open3d as o3d
from PIL import Image
import numpy as np

# Ensure the repository root is on sys.path so we can import `utils` from anywhere
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.utils import compute_depth, load_config, get_calibration_values, transform_image

""""
This script runs a depth estimation model on a directory of RGB images and saves the depth images.
"""

# LOAD CONFIG
CONFIG_PATH = os.path.join(repo_root, "config.yml")
config = load_config(CONFIG_PATH)

def _latest_data_dir(prefix):
    prefix = os.path.normpath(prefix)
    directory_parent = os.path.dirname(prefix)
    basename = os.path.basename(prefix)
    if not basename:
        raise ValueError(f"save_dir_prefix must have a basename, got '{prefix}'")

    if os.path.isabs(directory_parent):
        parent_dir = directory_parent
    elif directory_parent:
        parent_dir = os.path.join(repo_root, directory_parent)
    else:
        parent_dir = repo_root

    parent_dir = os.path.abspath(parent_dir)
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(f"parent data directory '{parent_dir}' not found")

    candidates = [
        os.path.join(parent_dir, entry)
        for entry in os.listdir(parent_dir)
        if entry.startswith(basename) and os.path.isdir(os.path.join(parent_dir, entry))
    ]
    if not candidates:
        raise FileNotFoundError(f"no data directories starting with '{basename}' in '{parent_dir}'")

    latest_dir = max(candidates, key=os.path.getmtime)
    print(f"[estimate_depth] using latest data directory: {latest_dir}", flush=True)
    return latest_dir

data_dir = config.get("data_dir")
if not data_dir:
    data_dir = _latest_data_dir(config["save_dir_prefix"])
camera_source = config["camera_source"] # what camera was used for the RGB images?
print("Loading" + camera_source + "images from: ", data_dir, ".")

# Set & create directories for images
rgb_dir = os.path.join(data_dir, camera_source + "-rgb-images")
kinect_img_dir = os.path.join(data_dir, "kinect-rgb-images")
os.mkdir(kinect_img_dir) if not os.path.exists(kinect_img_dir) else None
kinect_depth_dir = os.path.join(data_dir, "kinect-depth-images")
os.mkdir(kinect_depth_dir) if not os.path.exists(kinect_depth_dir) else None
print("Saving Depth images to: ", kinect_depth_dir)

# Load the calibration values
camera_calibration_path = config["camera_calibration_path"]
mtx, dist, optimal_mtx, roi = get_calibration_values(camera_calibration_path)
# Kinect intrinsic matrix (PrimeSense defaults roughly match kinect)
kinect = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# Load the ZoeDepth model
# conf = get_config("zoedepth", config["zoedepth_mode"]) # NOTE: "eval" runs slightly slower, but is stated to be more metrically accurate
# model_zoe = build_model(conf)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device is: ", DEVICE)
# zoe = model_zoe.to(DEVICE)

# Figure out how many images are in folder by counting .jpg files
end_frame = len([name for name in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, name)) and name.endswith(".jpg")])
end_frame = end_frame - 1 # Ignore last image since sometimes pose information is not saved for it

start_time = time.time()

for frame_number in range(0, end_frame):
    print("Applying ZoeDepth to:  %d/%d"%(frame_number+1,end_frame))
    filename = rgb_dir + "/" + camera_source + "_frame-%06d.rgb.jpg"%(frame_number)
    # Read in image with Pillow and convert to RGB
    crazyflie_rgb = Image.open(filename)#.convert("RGB")  # load
    # Resize, undistort (optional), and warp image to kinect's dimensions and intrinsics
    kinect_rgb = transform_image(
        np.asarray(crazyflie_rgb),
        mtx,
        dist,
        kinect,
        roi,
        apply_undistort=config.get("enable_undistort", True),
    )
    kinect_rgb = cv2.cvtColor(kinect_rgb, cv2.COLOR_BGR2RGB)
    # Compute depth
    depth_numpy, depth_colormap = compute_depth(kinect_rgb, zoe)
    # Save images
    cv2.imwrite(kinect_img_dir + "/kinect_frame-%06d.rgb.jpg"%(frame_number), kinect_rgb)
    cv2.imwrite(kinect_depth_dir + "/" + "kinect_frame-%06d.depth.jpg"%(frame_number), depth_colormap)
    np.save(kinect_depth_dir + "/" + "kinect_frame-%06d.depth.npy"%(frame_number), depth_numpy) # saved in meters

print("Time to compute depth for %d images: %f"%(end_frame, time.time()-start_time))
