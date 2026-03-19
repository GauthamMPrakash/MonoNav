r"""
  __  __                   _   _             
 |  \/  | ___  _ __   ___ | \ | | __ ___   __
 | |\/| |/ _ \| '_ \ / _ \|  \| |/ _` \ \ / /
 | |  | | (_) | | | | (_) | |\  | (_| |\ V / 
 |_|  |_|\___/|_| |_|\___/|_| \_|\__,_| \_/  
Copyright (c) 2023 Nate Simon
License: MIT
Authors: Nate Simon and Anirudha Majumdar, Princeton University
Project Page: https://natesimon.github.io/mononav

The purpose of this script is to fuse depth images and poses into a 3D reconstruction.
Here, we use Open3D's tensor reconstruction system: the VoxelBlockGrid.

After fusion, the reconstruction is visualized (in addition to the camera poses), and saved to file.

"""
addPose = True  # Visualize camera poses in addition to the point cloud
data_dir = None # if None, will automatically look for latest data directory with prefix specified in config.yml (e.g. "data/flight-")

import numpy as np
import time
import os
import sys
import open3d as o3d
from PIL import Image
import numpy as np
import yaml

# Ensure the repository root is on sys.path so we can import `utils` from anywhere
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.utils import *
#####################################################################

CONFIG_PATH = "../config.yml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

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
    print(f"[fuse_depth] using latest data directory: {latest_dir}", flush=True)
    return latest_dir

if data_dir is None:
    data_dir = _latest_data_dir(config["save_dir_prefix"])  # parent directory to look for RGB images, and save depth images
rgb_dir = os.path.join(data_dir, "rgb-images")
depth_dir = os.path.join(data_dir, "transform-depth-images")
pose_dir = os.path.join(data_dir, "poses")
#####################################################################

# Initialize TSDF VoxelBlockGrid
depth_scale = config["VoxelBlockGrid"]["depth_scale"]
depth_max = config["VoxelBlockGrid"]["depth_max"]
trunc_voxel_multiplier = config["VoxelBlockGrid"]["trunc_voxel_multiplier"]
weight_threshold = config["weight_threshold"] # for planning and visualization (!! important !!)
if config['VoxelBlockGrid']['device'] == "None":
    import torch
    device = 'CUDA:0' if torch.cuda.is_available() else 'CPU:0'
else:
    device = config['VoxelBlockGrid']['device']

vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device))
        
#####################################################################

poses = [] # for visualization
t_start = time.time()

depth_files = [name for name in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, name)) and name.endswith(".jpg")]
depth_files = sorted(depth_files)

# Get last frame
first_frame = split_filename(depth_files[0])
end_frame = split_filename(depth_files[-1])

for filename in depth_files:
    # Get the frame number from the depth filename
    frame_number = split_filename(filename)
    print("Integrating frame %d/%d"%(frame_number,end_frame))
    rgb_file = os.path.join(rgb_dir, "frame-%06d.rgb.jpg"%frame_number)

    # Read in camera pose
    pose_file = os.path.join(pose_dir, "frame-%06d.pose.txt"%frame_number)
    cam_pose = np.loadtxt(pose_file)
    poses.append(cam_pose)

    # Get color image with Pillow and convert to RGB
    color = Image.open(rgb_file).convert("RGB")  # load

    # Integrate
    depth_file = os.path.join(depth_dir, "transform_frame-%06d.depth.npy"%frame_number)
    depth_numpy = np.load(depth_file) # mm
    vbg.integration_step(color, depth_numpy, cam_pose)

#####################################################################
# Print out timing information
t_end = time.time()
print("Time taken (s): ", t_end - t_start)
print("FPS: ", end_frame/(t_end - t_start))

pcd = vbg.vbg.extract_point_cloud(weight_threshold)

if addPose:
    pose_lineset = get_poses_lineset(poses)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd.to_legacy())
    visualizer.add_geometry(pose_lineset)
    for pose in poses:
        # Add coordinate frame ( The x, y, z axis will be rendered as red, green, and blue arrows respectively.)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.5, center=(0, 0, 0))
        visualizer.add_geometry(coordinate_frame.transform(pose))
    visualizer.run()
    visualizer.destroy_window()
else:
    o3d.visualization.draw([pcd])

#####################################################################

npz_filename = os.path.join(data_dir, "vbg.npz")
ply_filename = os.path.join(data_dir, "pointcloud.ply")
print('Saving npz to {}...'.format(npz_filename))
print('Saving ply to {}...'.format(ply_filename))

vbg.vbg.save(npz_filename)
o3d.io.write_point_cloud(ply_filename, pcd.to_legacy())

print('Saving finished')
