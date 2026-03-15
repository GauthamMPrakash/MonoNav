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

# model parameters read from configuration
INPUT_SIZE = config['INPUT_SIZE']      # image scale parameter passed to compute_depth
CHECKPOINT = config['DA2_CHECKPOINT']  # path to checkpoint for DepthAnythingV2
# The encoder is now explicitly set in the config; fall back to parsing the
# checkpoint filename if the field is missing for backwards compatibility.
ENCODER = CHECKPOINT[-8:-4]  # crude parse: expects filenames like "..._vits.pth", "..._vitb.pth", etc.
if ENCODER is None:
    ENCODER = CHECKPOINT.split('_')[-1].split('.')[0]  # crude parse: last part before extension
    print(f"[warning] DA2_ENCODER not set in config, parsed '{ENCODER}' from checkpoint name")
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

print(f"[device] torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
print(f"[device] selected DEVICE={DEVICE}, model_device={model_device}", flush=True)
if torch.cuda.is_available():
    print(f"[device] cuda_name={torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
if model_device.type != 'cuda' and torch.cuda.is_available():
    print("[warning] CUDA is available but model is not on CUDA.", flush=True)

# GLOBAL VARIABLES
last_key_pressed = None  # store the last key pressed
shouldStop = False
trajectory_execution_stop_event = threading.Event()
traj_index = None
trajectory_start_time = None  # when the current trajectory started
autonomous_mode = False

# Intrinsics for undistort (optional)
camera_calibration_path = config.get('camera_calibration_path')
enable_undistort = config.get('enable_undistort', True)
if enable_undistort and camera_calibration_path:
    mtx, dist, optimal_mtx, roi = get_calibration_values(camera_calibration_path) # for the robot's camera
    calib_width, calib_height = get_calibration_resolution(camera_calibration_path)
    # only compute cropped intrinsics if roi is valid
    if roi is not None:
        fusion_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
    else:
        fusion_intrinsics = None
else:
    # no calibration available or undistort disabled; use defaults
    mtx = dist = optimal_mtx = roi = None
    calib_width = calib_height = None
    fusion_intrinsics = None  # VoxelBlockGrid will choose default intrinsics

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
max_traj_idx = None  # Start with no trajectory; only set when user presses 'g' for MonoNav mode
print(f"Trajectory library loaded: {len(traj_list)} trajectories", flush=True)
print(f"Trajectory amplitudes: left(idx 0)={amplitudes[0]:.3f}, right(idx {len(amplitudes)-1})={amplitudes[-1]:.3f}", flush=True)
print("Press 'g' to enable MonoNav autonomous mode, or 'a'/'w'/'d' for manual left/straight/right", flush=True)

# Planning presets
filterYvals = config['filterYvals']
filterWeights = config['filterWeights']
filterTSDF = config['filterTSDF']
if 'goal_position_rdf' in config:
    # Goal is in RDF meters: [right, down, forward]
    goal_position = np.array(config['goal_position_rdf'])
else:
    goal_position = None # non-directed exploration

print("Goal position (RDF): ", goal_position, flush=True)
min_dist2obs = config['min_dist2obs']
min_dist2goal = config['min_dist2goal']
print(f"Trajectory index convention: 0=sharp left, {len(traj_list)//2}=straight, {len(traj_list)-1}=sharp right", flush=True)

# Make directories for data
time_string = time.strftime('%Y-%m-%d-%H-%M-%S')
save_dir = config['save_dir_prefix'] + time_string
print("Saving files to: " + save_dir, flush=True)
npz_save_filename = save_dir + '/vbg.npz'

img_dir = os.path.join(save_dir, 'rgb-images')
pose_dir = os.path.join(save_dir, 'poses')
transform_img_dir = os.path.join(save_dir, 'transform-rgb-images')
transform_depth_dir = os.path.join(save_dir, 'transform-depth-images')

# Ensure the base save directory exists so file writes succeed
os.makedirs(save_dir, exist_ok=True)

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
        if key.char is not None:
            last_key_pressed = key.char.lower()
    except AttributeError:
        # Ignore non-character keys in this controller.
        pass

# Fallback key capture from OpenCV window (useful when pynput can't access keyboard,
# e.g., running with sudo or headless input devices).
def update_key_from_cv(wait_ms=1):
    global last_key_pressed
    k = cv2.waitKey(wait_ms) & 0xFF
    if k != 255:
        try:
            last_key_pressed = chr(k).lower()
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
    global autonomous_mode

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

    # Only update VBG intrinsics when valid calibration intrinsics are available.
    if optimal_mtx is not None and roi is not None:
        vbg_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
        vbg.intrinsic_matrix = vbg_intrinsics
        vbg.depth_intrinsic = o3d.core.Tensor(vbg_intrinsics)

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
        mavc.en_pose_stream()                    # Commands AP to stream poses at a deafult value of 15 Hz
        hdg = mavc.get_pose()[3] # get initial yaw
        if FLY_VEHICLE==True:
            print("Arming Motors!", flush=True)
            mavc.set_mode('GUIDED')
            mavc.arm()
            print("Taking off.", flush=True)
            mavc.takeoff(height)
            # mavc.set_speed(forward_speed)
        
        start_pose_thread()                      # Start background pose polling at 10 Hz (non-blocking)
        
        # Convert RDF goal to NED, then reorder to internal [E, D, N]
        # to match camera_position[0:-1, -1] from get_pose_matrix().
        if goal_position is not None:
            goal_position = np.array(
                rdf_goal_to_ned(goal_position[0], goal_position[1], goal_position[2], hdg)
            )
            print(f"Goal position (NED): {goal_position}", flush=True)
            goal_position = np.array([goal_position[1], goal_position[2], goal_position[0]]).reshape(1, 3)
        mavc.printd(f"Heading offset : {hdg*180/np.pi}")
    
        print("Starting control.", flush=True)
        traj_counter = 0         # how many trajectory iterations have we done?
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
            
            # Check for stop keys first (these exit the control loop)
            if last_key_pressed == 'q':
                autonomous_mode = False
                mavc.eSTOP()
                print("Pressed q. EMERGENCY STOP.", flush=True)
                shouldStop = True
                break
            elif last_key_pressed == 'c':
                autonomous_mode = False
                mavc.set_mode('LAND')
                print("Pressed c. Ending control.", flush=True)
                shouldStop = True
                break
            elif last_key_pressed == 'r':
                autonomous_mode = False
                print("\nPressed r. Switching to SmartRTL.\n", flush=True)
                if FLY_VEHICLE:
                    mavc.set_mode('SmartRTL')
                shouldStop = True
                break
            
            # Check for mode/command keys (these change behavior but don't stop)
            if last_key_pressed == 'h':
                autonomous_mode = False
                print("Pressed h. Hovering in place.", flush=True)
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0) # hover in place
                traj_index = None  # Stop following any trajectory
                last_key_pressed = None
            
            # Check for trajectory control keys
            if last_key_pressed == 'a':
                autonomous_mode = False
                print("Pressed a. Going left.", flush=True)
                traj_index = 0 # left
                print(f"[main] set traj_index={traj_index} (manual)", flush=True)
                last_key_pressed = None
            elif last_key_pressed == 'w':
                autonomous_mode = False
                print("Pressed w. Going straight.", flush=True)
                traj_index = len(traj_list)//2 # straight
                print(f"[main] set traj_index={traj_index} (manual)", flush=True)
                last_key_pressed = None
            elif last_key_pressed == 'd':
                autonomous_mode = False
                print("Pressed d. Going right.", flush=True)
                traj_index = len(traj_list)-1 # right
                print(f"[main] set traj_index={traj_index} (manual)", flush=True)
                last_key_pressed = None
            elif last_key_pressed == 'g':
                autonomous_mode = True
                print("Pressed g. Using MonoNav.", flush=True)
                    # Keep GO mode active, but execute a hover-only cycle and re-plan.
                if max_traj_idx is not None:
                    traj_index = max_traj_idx
                    print(f"[main] set traj_index={traj_index} (from max_traj_idx on g)", flush=True)
                last_key_pressed = None
        
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
                pose = mavc.get_pose()
                camera_position = get_pose_matrix(*pose)

                if goal_position is not None:
                    dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
                    if dist_to_goal <= min_dist2goal:
                        print("Reached goal!")
                        autonomous_mode = False
                        break

                # Optionally transform camera image (undistort + crop) based on config
                if config.get("enable_undistort", True):
                    transform_bgr = transform_image(bgr, mtx, dist, optimal_mtx, roi)
                    transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)
                else:
                    transform_bgr = bgr
                    transform_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                #transform_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # compute depth
                depth_numpy, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE, make_colormap=True)

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

        # In autonomous GO mode, update selected trajectory from planner.
            if autonomous_mode:
                # In 'g' mode: run trajectory planning
                max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)
                if max_traj_idx is None:
                    if FLY_VEHICLE:
                        mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0)
                        time.sleep(0.2)
                    traj_none_count += 1
                    if traj_none_count > traj_max_none_count:
                        print("[INFO] No safe trajectory. Hovering in place.", flush=True)
                    time.sleep(1)
                else:
                    traj_none_count = 0
                    print(f"[TRAJ] Selected traj: {max_traj_idx}/{len(traj_list)-1}", flush=True)
                    traj_index = max_traj_idx

    # Exited while(!shouldStop); end control!
        # print("shouldStop: ", shouldStop)
        # print(f"[INFO] {end_message}")
        print("[INFO] End control.", flush=True)
        print("[INFO] Current distance to goal (m): ", np.linalg.norm(camera_position[0:-1, -1]-goal_position) if goal_position is not None else "N/A", flush=True)
        camera_position = [camera_position[0:-1, -1][2], camera_position[0:-1, -1][0], camera_position[0:-1, -1][1]] # print in NED order for readability
        print("[INFO] Current NED coords:" , *camera_position, flush=True)
        print("[INFO] Current RDF coords:", *ned_to_rdf(camera_position[0], camera_position[1], camera_position[2], hdg), flush=True)

        # save and view vbg (robust to missing save dir; always attempt visualization)
        try:
            save_dir = os.path.dirname(npz_save_filename) if npz_save_filename else None
            if save_dir and os.path.isdir(save_dir):
                print("Saving to {}...".format(npz_save_filename), flush=True)
                try:
                    vbg.vbg.save(npz_save_filename)
                    print("Saving finished", flush=True)
                except Exception as e:
                    print(f"[warning] failed to save VoxelBlockGrid: {e}", flush=True)
            else:
                print("Save directory not present or saving disabled; skipping VBG file save", flush=True)
        except Exception as e:
            print(f"[warning] exception while attempting to save VBG: {e}", flush=True)

        # Attempt to extract and visualize point cloud regardless of save success
        try:
            print("Visualize raw pointcloud.", flush=True)
            pcd = vbg.vbg.extract_point_cloud(weight_threshold)
            pcd_cpu = pcd.cpu()
            # Convert tensor point cloud to legacy for reliable visualization
            pcd_legacy = pcd_cpu.to_legacy()
            print(f"Point cloud has {len(pcd_legacy.points)} points", flush=True)
            visualize_pointcloud(pcd_legacy, window_name="MonoNav Reconstruction")
        except Exception as e:
            print(f"[warning] failed to extract/visualize point cloud: {e}", flush=True)

    except KeyboardInterrupt:
        mavc.set_mode('LAND')
        shouldStop = True
        print("\n[INTERRUPT] Ctrl+C detected. Sent land command.", flush=True)
            
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
