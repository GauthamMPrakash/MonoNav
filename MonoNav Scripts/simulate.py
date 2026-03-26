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

The purpose of this script is to step through the 3D reconstruction and to execute the MonoNav planner.
Steps:
1) load the reconstruction, poses, and trajectory library,
2) for each pose, choose the optimal motion primitive according to the planner,
3) visualize the reconstruction, poses, and motion primitives (both available and chosen).

This script is a useful way to debug and debrief the planner, as well as to see how changes to the planner
and trajectory library affect the planning performance.

"""

import os
import open3d as o3d
import numpy as np
import copy
import sys

# Ensure the repository root is on sys.path so we can import `utils` from anywhere
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.utils import load_config, poses_from_posedir, get_poses_lineset, get_trajlist, get_traj_linesets, choose_primitive

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
    print(f"[simulate] using latest data directory: {latest_dir}", flush=True)
    return latest_dir

config = load_config(os.path.join(repo_root, "config.yml"))
data_dir = config.get("data_dir")
if not data_dir:
    data_dir = _latest_data_dir(config["save_dir_prefix"])

pose_dir = os.path.join(data_dir, "poses")
trajlib_dir = os.path.join(repo_root, config.get("trajlib_dir", "utils/trajlib"))

# Load the VoxelBlockGrid from file.
files = [file for file in os.listdir(data_dir) if file.endswith('.npz')]
assert len(files) > 0, "No *.npz files found."
npz_filename = files[0] # if there are multiple files, change the index
print("Loading ", npz_filename, " with Open3D.")
vbg = o3d.t.geometry.VoxelBlockGrid.load(os.path.join(data_dir, npz_filename)).cpu()
pcd = vbg.extract_point_cloud(config["weight_threshold"])

# Planning presets
filterYvals = config["filterYvals"]
filterWeights = config["filterWeights"]
filterTSDF = config["filterTSDF"]
if "goal_position" in config:
    goal_position = np.array(config["goal_position"]).reshape(1, 3) # OpenCV frame: +X RIGHT, +Y DOWN, +Z FORWARD
else:
    goal_position = None
print("Goal position: ", goal_position)
min_dist2obs = config["min_dist2obs"]
weight_threshold = config["weight_threshold"] # for planning and visualization

# Load poses from directory.
poses =  poses_from_posedir(pose_dir)
# Get pose lineset
pose_lineset = get_poses_lineset(poses)

# Load the trajectory linesets from the trajlib directory
traj_list = get_trajlist(trajlib_dir)
traj_linesets, period, forward_speed, amplitudes = get_traj_linesets(traj_list)


# Create the visualizer and add components
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(pcd.to_legacy())
visualizer.add_geometry(pose_lineset)

# For each pose, compute the optimal motion primitive.
# Paint the optimal motion primitive green, and the rest black.
n = 5 # iterate over every n poses
for i in range(0, len(poses), n):
    pose = poses[i]
    max_traj_idx = choose_primitive(vbg, pose, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)
    for traj_idx, traj_lineset in enumerate(traj_linesets):
        traj_lineset_copy = copy.deepcopy(traj_lineset)
        traj_lineset_copy.transform(pose)

        if traj_idx == max_traj_idx:
            traj_lineset_copy.paint_uniform_color([0, 1, 0])
        else:
            traj_lineset_copy.paint_uniform_color([0, 0, 0])

        visualizer.add_geometry(traj_lineset_copy)

    # # (Optional) Uncomment to add coordinate frame, which may look busy.
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.5, center=(0, 0, 0))
    # visualizer.add_geometry(coordinate_frame.transform(pose))

visualizer.run()
visualizer.destroy_window()
