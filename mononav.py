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
import threading

# Add DepthAnythingV2-metric path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
metric_depth_path = os.path.join(repo_root, 'DepthAnythingV2-metric')
sys.path.insert(0, metric_depth_path)
from depth_anything_v2.dpt import DepthAnythingV2

import utils.mavlink_control as mavc   # import the mavlink helper script          

# Helper functions. Core MonoNav algorithms are implemented in utils.py
from utils.utils import *
from pynput import keyboard            # Keyboard control

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
STREAM_URL = config['camera_ip']       # ESP32 HTTP MJPEG stream
save_during_flight = config['save_during_flight']

yawrate_gain = config['yawrate_gain']       # gain for yaw rate control (tuning parameter for your robot)
yvel_gain = config['yvel_gain']             # gain for lateral velocity to prevent sideslip during high velocity turns (probably not required for ArduPilot)

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
allow_smart_rtl = False   # only allow SmartRTL when goal reached or no safe trajectory
trajectory_execution_stop_event = threading.Event()
traj_index = None
trajectory_start_time = None  # when the current trajectory started

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
    device = 'CUDA:0' if o3d.core.cuda.is_available() else 'CPU:0'

vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device), intrinsic_matrix=fusion_intrinsics)

# Initialize Trajectory Library (Motion Primitives)
trajlib_dir = config['trajlib_dir']
traj_list = get_trajlist(trajlib_dir)
traj_linesets, period, forward_speed, amplitudes = get_traj_linesets(traj_list)
if config['forward_speed'] is not None:
    forward_speed = config['forward_speed']
max_traj_idx = int(len(traj_list)/2) # set initial value to that of FORWARD flight (should be median value)
print("Initial trajectory chosen: %d out of %d"%(max_traj_idx, len(traj_list)))
print(f"Trajectory amplitudes: left(idx 0)={float(amplitudes[0]):.3f}, right(idx {len(amplitudes)-1})={float(amplitudes[-1]):.3f}")

# Planning presets
filterYvals = config['filterYvals']
filterWeights = config['filterWeights']
filterTSDF = config['filterTSDF']
if 'goal_position_rdf' in config:
    # Goal is in RDF meters: [right, down, forward]
    goal_position = np.array(config['goal_position_rdf'])
else:
    goal_position = None # non-directed exploration

print("Goal position (RDF): ", goal_position)
min_dist2obs = config['min_dist2obs']
min_dist2goal = config['min_dist2goal']
print(f"Trajectory index convention: 0=sharp left, {len(traj_list)//2}=straight, {len(traj_list)-1}=sharp right")

# Make directories for data
time_string = time.strftime('%Y-%m-%d-%H-%M-%S')
save_dir = config['save_dir_prefix'] + time_string
print("Saving files to: " + save_dir)
npz_save_filename = save_dir + '/vbg.npz'

img_dir = os.path.join(save_dir, 'rgb-images')
pose_dir = os.path.join(save_dir, 'poses')
transform_img_dir = os.path.join(save_dir, 'transform-rgb-images')
transform_depth_dir = os.path.join(save_dir, 'transform-depth-images')

if save_during_flight:
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

# Trajectory execution thread: sends motion commands at fixed rate
def trajectory_execution_loop(command_hz=10):
    """Background thread: execute trajectory with smooth velocity commands at fixed rate."""
    global traj_index, trajectory_start_time
    period_s = 1.0 / max(command_hz, 1)
    next_command = time.monotonic()
    
    # log whenever the execution loop begins iteration (for debugging thread activity)
    while not trajectory_execution_stop_event.is_set():
        # (traj_index may be None until a trajectory is selected)
        if traj_index is not None and trajectory_start_time is not None:
            elapsed = time.time() - trajectory_start_time
            
            # Check if we've exceeded the trajectory period
            if elapsed < period:
                # Compute smooth yaw rate for this instant
                yawrate = amplitudes[traj_index] * np.sin(np.pi / period * elapsed)  # rad/s
                yvel = yawrate * yvel_gain
                yawrate = yawrate * yawrate_gain
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(forward_speed, yvel, yaw_rate=yawrate)
        
        next_command += period_s
        sleep_time = next_command - time.monotonic()
        if sleep_time > 0:
            trajectory_execution_stop_event.wait(sleep_time)
        else:
            next_command = time.monotonic()

# MAIN MONONAV CONTROL LOOP
def main():
    global shouldStop
    global last_key_pressed
    global max_traj_idx
    global traj_index
    global trajectory_start_time
    global mtx
    global dist
    global optimal_mtx
    global roi
    global goal_position
    global traj_max_none_count

    traj_none_count = 0
    traj_max_none_count = 2                      # how many times traj chooser can predict no safe traj before we force stop

    while True:
        # ARDUCOPTER CONTROL
        # Connect to the drone
        mavc.connect_drone(IP, baud=baud)
        mavc.set_ekf_origin(EKF_LAT, EKF_LON, 0) # Ignored if already close to the previous origin, if set
        reboot = mavc.reboot_if_EKF_origin(0.5)  # Call this function after enabling pose_stream
        if reboot:
            mavc.printd("Rebooted drone to set EKF origin. Waiting for reconnection...")
            time.sleep(7)    # Wait for drone to reboot
            mavc.printd("Reconnecting...")
            continue         # Restart connection loop
        break

    # Run the depth model a few times (the first inference is slow), and skip the first few frames
    cap = VideoCapture(STREAM_URL)
    first_bgr = cap.read()
    frame_height, frame_width = first_bgr.shape[:2]

    if calib_width is not None and calib_height is not None:
        mtx, dist, optimal_mtx, roi = adjust_intrinsics_to_frame_size(
            mtx, dist, optimal_mtx, roi, frame_width, frame_height, calib_width, calib_height
        )

    vbg_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
    vbg.intrinsic_matrix = vbg_intrinsics
    vbg.depth_intrinsic = o3d.core.Tensor(vbg_intrinsics, o3d.core.Dtype.Float64)

    # Scale mtx, dist to match current camera resolution
    # Read one frame to get actual resolution
    try:
        for i in range(0, config['num_pre_depth_frames']):
            bgr = cap.read()
            # COMPUTE DEPTH
            start_time_test = time.time()
            # depth_numpy, depth_colormap = compute_depth_fast(bgr, INPUT_SIZE)
            depth_numpy, depth_colormap = compute_depth(bgr, depth_anything, INPUT_SIZE)
            print("TIME TO COMPUTE DEPTH:", time.time() - start_time_test)
            cv2.imshow("test", bgr)
            update_key_from_cv()
        cv2.destroyAllWindows()

    # Initialize lists and frame counter.
        frame_number = 0
        start_flight_time = time.time()
        mavc.heading_offset_init()
        if FLY_VEHICLE==True:
            print("Arming Motors!")
            mavc.set_mode('GUIDED')
            mavc.arm()
            print("Taking off.")
            mavc.takeoff(height)
            # mavc.set_speed(forward_speed)
        
        mavc.en_pose_stream()                    # Commands AP to stream poses at a deafult value of 15 Hz
        start_pose_thread()                      # Start background pose polling at 10 Hz (non-blocking)
        
        # Convert RDF goal to NED, then reorder to internal [E, D, N]
        # to match camera_position[0:-1, -1] from get_pose_matrix().
        if goal_position is not None:
            goal_position = np.array(
                rdf_goal_to_ned(goal_position[0], goal_position[1], goal_position[2], mavc.heading_offset),
                dtype=np.float64,
            )
            print(f"Goal position (NED): {goal_position}")
            goal_position = np.array([goal_position[1], goal_position[2], goal_position[0]], dtype=np.float64).reshape(1, 3)
        mavc.printd(f"Heading offset : {mavc.heading_offset*180/np.pi}")
    
        print("Starting control.")
        traj_counter = 0         # how many trajectory iterations have we done?
        no_safe_traj = False
        last_time = time.time()  # for FPS counter
        
        # Start trajectory execution thread
        trajectory_execution_stop_event.clear()
        traj_thread = threading.Thread(
            target=trajectory_execution_loop,
            args=(10,),  # 10 Hz command rate
            daemon=True,
        )
        traj_thread.start()
    #   start_time = time.time() # seconds

        while not shouldStop:
            update_key_from_cv(1)
            if last_key_pressed == 'a':
                print("Pressed a. Going left.")
                traj_index = 0 # left
            elif last_key_pressed == 'w':
                print("Pressed w. Going straight.")
                traj_index = len(traj_list)//2 # straight
            elif last_key_pressed == 'd':
                print("Pressed d. Going right.")
                traj_index = len(traj_list)-1 # right
            elif last_key_pressed == 'g':
                print("Pressed g. Using MonoNav.")
                if no_safe_traj:
                    if FLY_VEHICLE:
                        mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0)
                        print("No safe trajectory")
                    time.sleep(0.1)
                    continue

                traj_index = max_traj_idx

            elif last_key_pressed == 'h':
                print("Pressed h. Hovering in place.")
                mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0) # hover in place
                time.sleep(0.1)
                continue
            elif last_key_pressed == 'r':
                # only allow RTL if flagged by goal/no-safe-traj
                if allow_smart_rtl:
                    print("Pressed r. Switching to SmartRTL.")
                    if FLY_VEHICLE:
                        mavc.set_mode('SmartRTL')
                    shouldStop = True
                    break
                else:
                    last_key_pressed = 'g'
                    time.sleep(0.1)
                    continue
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
            if save_during_flight:
                traj_idx_to_log = max_traj_idx if max_traj_idx is not None else -1
                row = np.array([frame_number, int(traj_idx_to_log), time.time()-start_flight_time]) # time since start of flight
                with open(save_dir + '/trajectories.csv', 'a') as file:
                    np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')

        # Execute the selected trajectory in background thread.
        # Main thread now focuses on frame processing and TSDF integration.
            trajectory_start_time = time.time()
            traj_start = trajectory_start_time
            
            while time.time() - traj_start < period:
                bgr = cap.read()
            # get_latest_pose returns (x, y, z, yaw, pitch, roll) - non-blocking from thread
                pose = get_latest_pose()
                camera_position = get_pose_matrix(*pose)

                if goal_position is not None:
                    dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
                    if dist_to_goal <= min_dist2goal:
                        print("Reached goal!")
                        allow_smart_rtl = True
                        last_key_pressed = 'c'
                        break

                # Transform Camera Image to undistort and crop according to calibration
                transform_bgr = transform_image(bgr, mtx, dist, optimal_mtx, roi)
                transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)
                #transform_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # compute depth
                depth_numpy, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE)

                if depth_colormap is not None:
                    # Add FPS counter to depth display
                    processing_speed = 1 / (time.time() - last_time)
                    fps_text = ("%0.2f" % (processing_speed,)) + ' fps'
                    textsize = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.putText(
                        depth_colormap,
                        fps_text,
                        org=(int((depth_colormap.shape[1] - textsize[0] / 2)), int((textsize[1]) / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )
                    last_time = time.time()
                    cv2.imshow("Depth", depth_colormap)
                    update_key_from_cv(1)

            # integrate the vbg (prefers rgb)
                vbg.integration_step(transform_rgb, depth_numpy, camera_position)

                if save_during_flight:
                    cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg'%(frame_number), bgr)
                    cv2.imwrite(transform_img_dir + '/transform_frame-%06d.rgb.jpg'%(frame_number), transform_bgr)
                    cv2.imwrite(transform_depth_dir + '/' + 'transform_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
                    np.save(transform_depth_dir + '/' + 'transform_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
                    np.savetxt(pose_dir + '/frame-%06d.pose.txt'%(frame_number), camera_position)

                frame_number += 1
            traj_counter += 1

        # if not in "GO" (g) mode, reset to stopping mode
            if last_key_pressed != 'g':
                last_key_pressed = None

            shouldStop, max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)
            if max_traj_idx is None:
                traj_none_count += 1
                if traj_none_count > traj_max_none_count:
                    no_safe_traj = True
                    allow_smart_rtl = True
                print("[INFO] No safe trajectory. Hovering in place.")
                shouldStop = True
            else:
                no_safe_traj = False

            mavc.printd(f"SELECTED max_traj_idx: { max_traj_idx}")

    # Exited while(!shouldStop); end control!
        # print("shouldStop: ", shouldStop)
        # print(f"[INFO] {end_message}")
        print("[INFO] End control.")
        print("[INFO] Current distance to goal (m): ", np.linalg.norm(camera_position-goal_position[0]) if goal_position is not None else "N/A")
        camera_position = [camera_position[0:-1, -1][2], camera_position[0:-1, -1][0], camera_position[0:-1, -1][1]] # print in NED order for readability
        print("[INFO] Current NED coords:" , camera_position)
        print("[INFO] Current RDF coords:", ned_to_rdf(camera_position[0], camera_position[1], camera_position[2], mavc.heading_offset))

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
        visualize_pointcloud(pcd_legacy, window_name="MonoNav Reconstruction")

    except KeyboardInterrupt:
        mavc.set_mode('LAND')
        print("\n[INTERRUPT] Ctrl+C detected. Sent land command.")
            
    finally:
        # Stop trajectory execution thread
        trajectory_execution_stop_event.set()
        if 'traj_thread' in locals():
            traj_thread.join(timeout=2.0)
        
        print("Releasing camera capture.")
        cap.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
