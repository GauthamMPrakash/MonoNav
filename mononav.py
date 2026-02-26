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

This script runs the MonoNav navigation pipeline on an aerial vehicle with ArduCopter firmware.
Essentially, the collect_dataset, estimate_depth, fuse_depth, and simulate scripts are combined
into a single script that:
- Collects synchronized images and poses from the drone,
- Estimates depth using DepthAnythingV2,
- Fuses depth images and poses into a 3D reconstruction,
- Chooses a motion primitive according to the planner,
- Executes the motion primitive while collecting & fusing new images,
- Repeats until the goal is reached or no primitive satisfies the obstacle avoidance constraint.

This script is designed for an ArduCopter drone that is configured correctly as described in the ArduCopter setup instructions file and may require modification for your specific hardware!

"""

import cv2
import torch
import numpy as np
import time
import os
import open3d as o3d
import sys

# Add DepthAnythingV2-metric path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
metric_depth_path = os.path.join(repo_root, 'metric_depth')
sys.path.insert(0, metric_depth_path)
from depth_anything_v2.dpt import DepthAnythingV2

import mavlink_control as mavc         # import the mavlink helper script          
from pynput import keyboard            # Keyboard control

# helper functions
from utils.utils import *

# LOAD VALUES FROM CONFIG FILE
config = load_config('config.yml')

INPUT_SIZE = config['INPUT_SIZE']      # Image size
CHECKPOINT = config['DA2_CHECKPOINT']  # path to checkpoint for DepthAnythingV2
ENCODER = CHECKPOINT[-8:-4]            # extract encoder type from checkpoint filename (assumes format "DA2_{ENCODER}_checkpoint.pth")  
MAX_DEPTH = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IP = config['IP']
baud = config['baud']
height = config['height']
FLY_VEHICLE = config['FLY_VEHICLE']
EKF_LAT = config['EKF_LAT']
EKF_LON = config['EKF_LON']
STREAM_URL = config['camera_ip']       # YOUR ESP32 HTTP MJPEG stream

# DepthAnythingV2 model configurations. You typically only need small or base models
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

print(f"[device] torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"[device] selected DEVICE={DEVICE}, model_device={model_device}")
if torch.cuda.is_available():
    print(f"[device] cuda_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
if model_device.type != 'cuda' and torch.cuda.is_available():
    print("[warning] CUDA is available but model is not on CUDA.")

# GLOBAL VARIABLES
last_key_pressed = None  # store the last key pressed
shouldStop = False

# Intrinsics for undistortion
camera_calibration_path = config['camera_calibration_path']
mtx, dist, optimal_mtx, roi = get_calibration_values(camera_calibration_path) # for the robot's camera
calib_width, calib_height = get_calibration_resolution(camera_calibration_path)
fusion_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
# kinect = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault) # for the kinect

# Initialize VoxelBlockGrid
depth_scale = config['VoxelBlockGrid']['depth_scale']
depth_max = config['VoxelBlockGrid']['depth_max']
trunc_voxel_multiplier = config['VoxelBlockGrid']['trunc_voxel_multiplier']
weight_threshold = config['weight_threshold'] # for planning and visualization (!! important !!)
if config['VoxelBlockGrid']['device'] != "None": 
    device = config['VoxelBlockGrid']['device']
else:
    device = 'CUDA:0' if torch.cuda.is_available() else 'CPU:0'

vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device), intrinsic_matrix=fusion_intrinsics)

# Initialize Trajectory Library (Motion Primitives)
trajlib_dir = config['trajlib_dir']
traj_list = get_trajlist(trajlib_dir)
traj_linesets, period, forward_speed, amplitudes = get_traj_linesets(traj_list)
max_traj_idx = int(len(traj_list)/2) # set initial value to that of FORWARD flight (should be median value)
print("Initial trajectory chosen: %d out of %d"%(max_traj_idx, len(traj_list)))

# Planning presets
filterYvals = config['filterYvals']
filterWeights = config['filterWeights']
filterTSDF = config['filterTSDF']
if 'goal_position' in config:
    # Negate right (+X, index 0) and forward (+Y, index 2) directions
    goal_position = np.array(config['goal_position'])
    goal_position[[0, 2]] *= -1
    goal_position = goal_position.reshape(1, 3)
else:
    goal_position = None # non-directed exploration
print("Goal position: ", goal_position)
min_dist2obs = config['min_dist2obs']
min_dist2goal = config['min_dist2goal']

# Make directories for data
time_string = time.strftime('%Y-%m-%d-%H-%M-%S')
save_dir = config['save_dir_prefix'] + time_string
print("Saving files to: " + save_dir)
npz_save_filename = save_dir + '/vbg.npz'

img_dir = os.path.join(save_dir, 'rgb-images')
pose_dir = os.path.join(save_dir, 'poses')
transform_img_dir = os.path.join(save_dir, 'transform-rgb-images')
transform_depth_dir = os.path.join(save_dir, 'transform-depth-images')

os.makedirs(img_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)
os.makedirs(transform_img_dir, exist_ok=True)
os.makedirs(transform_depth_dir, exist_ok=True)

# Save the run information to a csv
header = ['frame_number', 'chosen_traj_idx', 'time_elapsed']
with open(save_dir + '/trajectories.csv', 'w') as file:
    file.write(','.join(header) + '\n')

# key press callback function (for manual control)
def on_press(key):
    global last_key_pressed
    try:
        last_key_pressed = key.char
    except AttributeError:
        last_key_pressed = key

# Fallback key capture from OpenCV window (useful when pynput can't access keyboard,
# e.g., running with sudo or headless input devices).
def update_key_from_cv(wait_ms=1):
    global last_key_pressed
    k = cv2.waitKey(wait_ms) & 0xFF
    if k != 255:
        try:
            last_key_pressed = chr(k)
        except ValueError:
            pass
# start keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# MAIN MONONAV CONTROL LOOP
def main():
    global shouldStop
    global last_key_pressed
    global max_traj_idx
    global mtx
    global optimal_mtx
    global roi

    # Run the depth model a few times (the first inference is slow), and skip the first few frames
    cap = VideoCapture(STREAM_URL)
    first_bgr = cap.read()
    frame_height, frame_width = first_bgr.shape[:2]

    if calib_width is not None and calib_height is not None:
        scale_x = frame_width / calib_width
        scale_y = frame_height / calib_height
        if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
            mtx = scale_intrinsics(mtx, scale_x, scale_y)
            optimal_mtx = scale_intrinsics(optimal_mtx, scale_x, scale_y)
            roi = np.array(
                [
                    int(round(roi[0] * scale_x)),
                    int(round(roi[1] * scale_y)),
                    int(round(roi[2] * scale_x)),
                    int(round(roi[3] * scale_y)),
                ],
                dtype=np.int32,
            )

    vbg_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
    vbg.intrinsic_matrix = vbg_intrinsics
    vbg.depth_intrinsic = o3d.core.Tensor(vbg_intrinsics, o3d.core.Dtype.Float64)

    # Scale mtx, dist to match current camera resolution
    # Read one frame to get actual resolution
    try:
        for i in range(0, config['num_pre_depth_frames']):
            bgr = first_bgr if i == 0 else cap.read()
            # COMPUTE DEPTH
            start_time_test = time.time()
            depth_numpy, depth_colormap = compute_depth(bgr, depth_anything, INPUT_SIZE)
            print("TIME TO COMPUTE DEPTH:", time.time() - start_time_test)
            cv2.imshow("test", bgr)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        
    # ARDUCOPTER CONTROL
    # Connect to the drone
        mavc.connect_drone(IP, baud=baud)
        mavc.en_pose_stream()                    # Commands AP to stream poses at a deafult value of 15 Hz
        mavc.reboot_if_EKF_origin(0.3)           # Call this function after enabling pose_stream
        mavc.set_ekf_origin(EKF_LAT, EKF_LON, 0) # Ignored if already close to the previous origin, if set
    
    # Initialize lists and frame counter.
        frame_number = 0
        start_flight_time = time.time()
        if FLY_VEHICLE==True:
            print("Arming Motors!")
            mavc.set_mode('GUIDED')
            mavc.arm()
            print("Taking off.")
            mavc.takeoff(height)
            # mavc.set_speed(forward_speed)

    ##########################################
        print("Starting control.")
        traj_counter = 0         # how many trajectory iterations have we done?
        no_safe_traj = False
#   start_time = time.time() # seconds

        while not shouldStop:
            update_key_from_cv(1)
            if last_key_pressed == 'a':
                print("Pressed a. Going left.")
                traj_index = 0 # left
            elif last_key_pressed == 'w':
                print("Pressed w. Going straight.")
                traj_index = int(len(traj_list)/2) # straight
            elif last_key_pressed == 'd':
                print("Pressed d. Going right.")
                traj_index = len(traj_list)-1 # right
            elif last_key_pressed == 'g':
                print("Pressed g. Using MonoNav.")
                if no_safe_traj:
                    if FLY_VEHICLE:
                        mavc.send_body_offset_ned_vel(0, 0, 0, yaw_rate=0)
                        mavc.printd("No safe trajectory")
                    time.sleep(0.1)
                    continue
                traj_index = max_traj_idx
            elif last_key_pressed == 'c': #end control and land
                mavc.set_mode('LAND')
                print("Pressed c. Ending control.")
                shouldStop = True
                break
            elif last_key_pressed == 'q': #end flight immediately
                mavc.eSTOP()
                print("Pressed q. EMERGENCY STOP.")
                shouldStop = True
                break
            else:
                time.sleep(0.1)
                continue
        
        # Save trajectory information
            traj_idx_to_log = max_traj_idx if max_traj_idx is not None else -1
            row = np.array([frame_number, int(traj_idx_to_log), time.time()-start_flight_time]) # time since start of flight
            with open(save_dir + '/trajectories.csv', 'a') as file:
                np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')

        # Fly the selected trajectory, as applicable.
            start_time = time.time()            
            while time.time() - start_time < period:
                # WARNING: This controller is tuned for ArduCopter.
                # You must check whether your robot follows the open-loop trajectory.
                yawrate = amplitudes[traj_index]*np.sin(np.pi/period*(time.time() - start_time)) # rad/s
                yvel = yawrate*config['yvel_gain']
                yawrate = yawrate*config['yawrate_gain']
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(forward_speed, yvel, yaw_rate=yawrate)

            # get camera capture and transform intrinsics
                bgr = cap.read()
                #cv2.imshow("frame", bgr)
                update_key_from_cv(1)
                camera_position = get_drone_pose() # get camera position immediately
                if goal_position is not None:
                    dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
                    if dist_to_goal <= min_dist2goal:
                        print("Reached goal!")
                        shouldStop = True
                        last_key_pressed = 'c'
                        break
                # Transform Camera Image to Kinect Image
                transform_bgr = transform_image(np.asarray(bgr), mtx, dist, optimal_mtx, roi)
                transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)
                #transform_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # compute depth
                depth_numpy, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE)
                #depth_numpy, depth_colormap = compute_depth(bgr, depth_anything, INPUT_SIZE)
                cv2.imshow("frame", depth_colormap)

            # SAVE DATA TO FILE
                cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg'%(frame_number), bgr)
                cv2.imwrite(transform_img_dir + '/transform_frame-%06d.rgb.jpg'%(frame_number), transform_bgr)
                cv2.imwrite(transform_depth_dir + '/' + 'transform_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
                np.save(transform_depth_dir + '/' + 'transform_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
                np.savetxt(pose_dir + '/frame-%06d.pose.txt'%(frame_number), camera_position)

            # integrate the vbg (prefers rgb)
                vbg.integration_step(transform_rgb, depth_numpy, camera_position)

                frame_number += 1
            traj_counter += 1

        # if not in "GO" (g) mode, reset to stopping mode
            if last_key_pressed != 'g':
                last_key_pressed = None

            shouldStop, max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)
            if max_traj_idx is None:
                no_safe_traj = True
                shouldStop = True
                print("No safe trajectory. Hovering in place.")
            else:
                no_safe_traj = False
            print("SELECTED max_traj_idx: ", max_traj_idx)

    # Exited while(!shouldStop); end control!
        print("shouldStop: ", shouldStop)
        print("Reached goal OR too close to obstacles.")
        print("End control.")

        if FLY_VEHICLE:
            # Stopping sequence
            print("Landing.")
            mavc.set_mode('LAND')
            mavc.arm(0)

        # save and view vbg
        print("Saving to {}...".format(npz_save_filename))
        vbg.vbg.save(npz_save_filename)
        print("Saving finished")
        print("Visualize raw pointcloud.")
        pcd = vbg.vbg.extract_point_cloud(weight_threshold)
        pcd_cpu = pcd.cpu()
        # Convert tensor point cloud to legacy for reliable visualization
        pcd_legacy = pcd_cpu.to_legacy()
        print(f"Point cloud has {len(pcd_legacy.points)} points")
        if len(pcd_legacy.points) > 0:
        #    o3d.visualization.draw_geometries([pcd_legacy], window_name="MonoNav Reconstruction")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="MonoNav Reconstruction")
            vis.add_geometry(pcd_legacy)
            # Set a custom camera view (side or isometric)
            ctr = vis.get_view_control()
            # Set camera parameters: lookat, up, front, zoom
            # These values can be tuned for your data
            bounds = pcd_legacy.get_axis_aligned_bounding_box()
            center = bounds.get_center()
            ctr.set_lookat(center)
            ctr.set_up([0, 0, 1])  # Z up
            ctr.set_front([0, -1, 0])  # Look along -Y (forward), Z up
            ctr.set_zoom(0.7)
            vis.run()
            vis.destroy_window()

    finally:
        print("Releasing camera capture.")
        cap.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
