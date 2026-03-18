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
import traceback
from pynput import keyboard

# Add DepthAnythingV2-metric path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
metric_depth_path = os.path.join(repo_root, 'DepthAnythingV2-metric')
sys.path.insert(0, metric_depth_path)
from depth_anything_v2.dpt import DepthAnythingV2

import utils.mavlink_control as mavc   # import the mavlink helper script          

# Helper functions. Core MonoNav algorithms are implemented in utils.py
from utils.utils import *

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
vbg_device_cfg = config.get('VoxelBlockGrid', {}).get('device')
if vbg_device_cfg is None or str(vbg_device_cfg).lower() == "none":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    # Normalize for torch map_location (e.g., "CUDA:0" -> "cuda:0")
    DEVICE = str(vbg_device_cfg).lower()
IP = config['IP']
baud = config['baud']
height = config['height']
FLY_VEHICLE = config['FLY_VEHICLE']
EKF_LAT = config['EKF_LAT']
EKF_LON = config['EKF_LON']
STREAM_URL = config['camera_ip']       # ESP32 HTTP MJPEG stream
save_during_flight = config['save_during_flight']
enable_undistort = config.get('enable_undistort', True)

# Global variables
shouldStop = False
last_key_pressed = None
npz_save_filename = None
keyboard_listener = None

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
if vbg_device_cfg is None or str(vbg_device_cfg).lower() == "none":
    device = 'CUDA:0' if o3d.core.cuda.is_available() else 'CPU:0'
else:
    device_cfg_str = str(vbg_device_cfg)
    if device_cfg_str.lower().startswith('cuda'):
        suffix = device_cfg_str.split(':', 1)[1] if ':' in device_cfg_str else '0'
        device = f'CUDA:{suffix}'
    elif device_cfg_str.lower().startswith('cpu'):
        suffix = device_cfg_str.split(':', 1)[1] if ':' in device_cfg_str else '0'
        device = f'CPU:{suffix}'
    else:
        device = device_cfg_str

vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device), intrinsic_matrix=fusion_intrinsics)

# Initialize Trajectory Library (Motion Primitives)
trajlib_dir = config['trajlib_dir']
traj_list = get_trajlist(trajlib_dir)
traj_linesets, period, forward_speed, amplitudes = get_traj_linesets(traj_list)
if config['forward_speed'] is not None:
    forward_speed = config['forward_speed']
max_traj_idx = int(len(traj_list)/2)
print(f"\nTrajectory library loaded: {len(traj_list)} trajectories", flush=True)
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

min_dist2obs = config['min_dist2obs']
min_dist2goal = config['min_dist2goal']

if save_during_flight:
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
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(transform_img_dir, exist_ok=True)
    os.makedirs(transform_depth_dir, exist_ok=True)

    # Save the run information to a csv
    header = ['frame_number', 'chosen_traj_idx', 'time_elapsed']
    with open(save_dir + '/trajectories.csv', 'w') as file:
        file.write(','.join(header) + '\n')

def _on_key_press(key):
    """Capture keyboard input using pynput."""
    global last_key_pressed
    if key == keyboard.Key.esc:
        last_key_pressed = 'esc'
        return
    try:
        if key.char:
            last_key_pressed = key.char.lower()
    except AttributeError:
        return


def start_keyboard_listener():
    global keyboard_listener
    if keyboard_listener is None:
        keyboard_listener = keyboard.Listener(on_press=_on_key_press)
        keyboard_listener.start()


def stop_keyboard_listener():
    global keyboard_listener
    if keyboard_listener is not None:
        keyboard_listener.stop()
        keyboard_listener = None


def opencv_waitkey(mark_esc=True):
    """Process OpenCV GUI events; optionally latch Esc into global key state."""
    key = cv2.waitKey(1) & 0xFF
    if key == 27 and mark_esc:
        global last_key_pressed
        last_key_pressed = 'esc'
        return True
    return False

# MAIN MONONAV CONTROL LOOP
def main():
    global shouldStop
    global last_key_pressed
    global max_traj_idx
    global mtx, dist, optimal_mtx, roi
    global goal_position
    global save_dir

    cap = None
    camera_position = None
    traj_index = None # which primitive are we currently executing? (None for hover/no motion)

    while True:
        # ARDUCOPTER CONTROL
        # Connect to the drone
        print("Connecting to drone...", flush=True)
        mavc.connect_drone(IP, baud=baud)
        print("Connected.")
        mavc.en_pose_stream(30)                    # Commands AP to stream poses at given frequency
        mavc.set_ekf_origin(EKF_LAT, EKF_LON, 0) # Ignored if already close to the previous origin, if set
        reboot = mavc.reboot_if_EKF_origin(0.5)  # Call this function after enabling pose_stream
        if reboot:
            print("Rebooted drone to set EKF origin. Waiting for reconnection...")
            time.sleep(7)    # Wait for drone to reboot
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
            # Ignore warmup keys to avoid carrying stale ESC into the control loop.
            opencv_waitkey(mark_esc=False)
        cv2.destroyAllWindows()
        start_keyboard_listener()
        last_key_pressed = None
        shouldStop = False
        
        hdg = mavc.get_pose()[3] # get initial yaw
        # Convert startup nose-relative RDF goal to absolute NED, then reorder
        # to internal [E, D, N] so it matches camera_position[0:-1, -1].
        if goal_position is not None:
            print("\nGoal position (RDF): ", goal_position, flush=True)
            goal_position_ned = np.array(
                rdf_goal_to_ned(goal_position[0], goal_position[1], goal_position[2], hdg),
                dtype=float,
            )
            mavc.printd(f"Goal position (NED): {goal_position_ned}")
            goal_position = np.array(
                [goal_position_ned[1], goal_position_ned[2], goal_position_ned[0]],
                dtype=float,
            ).reshape(1, 3)
        mavc.printd(f"Heading offset: {hdg*180/np.pi}")
        mavc.set_mode('GUIDED')

        start_flight_time = time.time()
        if FLY_VEHICLE:
            print("Arming Motors!", flush=True)
            mavc.arm()
            print("Taking off.", flush=True)
            mavc.takeoff(height)
            # mavc.set_speed(forward_speed)
        start_pose_thread(15)                      # Start background pose polling at given frequency (non-blocking)
        print("Starting control.", flush=True)
        traj_counter = 0         # how many trajectory iterations have we done?
        # Initialize lists and frame counter.
        frame_number = 0
        processing_speed, loop_hz = 0, 0

        while not shouldStop:
            # Check for stop keys first (these exit the control loop)
            if last_key_pressed == 'esc':
                mavc.set_mode('BRAKE')
                mavc.set_mode('LAND')
                print("Pressed esc. Ending control.", flush=True)
                shouldStop = True
                last_key_pressed = None
                break
            elif last_key_pressed == 'p':
                """
                DO NOT USE FOR NORMAL FLYING. WILL STOP MOTORS IMMEDIATELY CAUSING A CRASH. Only use in emergency situations when you need to stop the drone immediately.
                """
                mavc.set_mode('BRAKE')
                mavc.arm(0)
                print("Pressed p. EMERGENCY STOP.", flush=True)
                shouldStop = True
                last_key_pressed = None
                break
            elif last_key_pressed == 'c':
                mavc.set_mode('BRAKE')
                mavc.set_mode('LAND')
                print("Pressed c. Ending control.", flush=True)
                shouldStop = True
                last_key_pressed = None
                break
            elif last_key_pressed == 'r':
                print("\nPressed r. Switching to SMART_RTL.\n", flush=True)
                if FLY_VEHICLE:
                    mavc.set_mode('SMART_RTL')
                shouldStop = True
                last_key_pressed = None
                break
            
            # Check for mode/command keys (these change behavior but don't stop)
            elif last_key_pressed == 'h':
                print("Pressed h. Hovering in place.", flush=True)
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0) # hover in place
                traj_index = None  # Stop following any trajectory
                last_key_pressed = None
            
            # Check for trajectory control keys
            if last_key_pressed == 'a':
                print("Pressed a. Going left.", flush=True)
                traj_index = 0 # left
                last_key_pressed = None
            elif last_key_pressed == 'w':
                print("Pressed w. Going straight.", flush=True)
                traj_index = len(traj_list)//2 # straight
                last_key_pressed = None
            elif last_key_pressed == 'd':
                print("Pressed d. Going right.", flush=True)
                traj_index = len(traj_list)-1  # right
                last_key_pressed = None
            elif last_key_pressed == 'g':      # GO mode
                print("Pressed g. Using MonoNav.", flush=True)
                traj_index = max_traj_idx
        
        # Save trajectory information
            if save_during_flight:
                traj_idx_to_log = max_traj_idx if max_traj_idx is not None else -1
                row = np.array([frame_number, int(traj_idx_to_log), time.time()-start_flight_time]) # time since start of flight
                with open(save_dir + '/trajectories.csv', 'a') as file:
                    np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')

            start_time = time.time()
            while time.time() - start_time <= period:
                frame_start_time = time.time()
                if traj_index is not None:
                    yawrate = amplitudes[traj_index] * np.sin(np.pi/period*(time.time() - start_time))  # rad/s
                    yvel = yawrate * yvel_gain
                    yawrate = yawrate * yawrate_gain
                    if FLY_VEHICLE:
                        mavc.send_body_offset_ned_vel(forward_speed, yvel, yaw_rate=yawrate)
                
                bgr = cap.read()
                # get_latest_pose returns (x, y, z, yaw, pitch, roll) - non-blocking from thread
                pose = get_latest_pose()

                # Optionally transform camera image (undistort + crop) based on config
                if enable_undistort:
                    transform_bgr = transform_image(bgr, mtx, dist, optimal_mtx, roi)
                    transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)
                else:
                    transform_bgr = bgr
                    transform_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                #transform_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # compute depth
                depth_numpy, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE, make_colormap=True)
                camera_position = get_pose_matrix(*pose)
                
                if last_key_pressed != 'h': # if not in hover mode, integrate into VBG
                    vbg.integration_step(transform_rgb, depth_numpy, camera_position)

                if goal_position is not None:
                    dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
                    if dist_to_goal < min_dist2goal:
                        print("Reached goal!")
                        shouldStop = True
                        break

                if depth_colormap is not None:
                    # time elapsed since the beginning of the current period
                    hz_text = ("%0.2f" % (loop_hz,)) + ' Hz '
                    fps_text = ("%0.2f" % (processing_speed,)) + ' FPS'
                    hz_size = cv2.getTextSize(hz_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    textsize = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

                    # Draw a second line of text above the FPS for additional info
                    # (no extra top margin; add a blank line underneath the Hz text)
                    hz_x = int((depth_colormap.shape[1] - hz_size[0] / 2))
                    hz_y = int(hz_size[1])
                    cv2.putText(
                        depth_colormap,
                        hz_text,
                        org=(hz_x, hz_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )

                    line_spacing = 5
                    fps_y = int(hz_y + hz_size[1] + line_spacing + textsize[1])

                    cv2.putText(
                        depth_colormap,
                        fps_text,
                        org=(int((depth_colormap.shape[1] - textsize[0] / 2)), int((textsize[1]) / 2 + hz_size[1] + 5)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255, 255, 255),
                    )
                    cv2.imshow("Depth", depth_colormap)
                #cv2.imshow("RGB", transform_bgr)
                if opencv_waitkey():
                    print("Pressed esc. Ending control.", flush=True)
                    shouldStop = True
                    last_key_pressed = 'c'
                    break

                if save_during_flight:
                    cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg'%(frame_number), bgr)
                    cv2.imwrite(transform_img_dir + '/transform_frame-%06d.rgb.jpg'%(frame_number), transform_bgr)
                    cv2.imwrite(transform_depth_dir + '/' + 'transform_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
                    np.save(transform_depth_dir + '/' + 'transform_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
                    np.savetxt(pose_dir + '/frame-%06d.pose.txt'%(frame_number), camera_position)

                frame_number += 1
            traj_counter += 1

            # Add FPS counter to depth display
            cur_time  = time.time()
            loop_hz = 1.0 / (cur_time - start_time)
            processing_speed = 1.0 / (cur_time - frame_start_time)

            # If this cycle ended due to a stop command, do not run planner updates.
            if shouldStop:
                break

            # In GO mode, update selected trajectory from planner.
            max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)

            mode = mavc.get_mode()
            if max_traj_idx is None:
                traj_index = None
                if mode == 'GUIDED':      
                    mavc.set_mode('BRAKE')
                    time.sleep(0.1) # brief brake before hover to prevent drift during stop
                print("[INFO] No safe trajectory found. Hovering in place.", flush=True)
            else:
                if mode == 'BRAKE':     # so that external mode commands from GCS or RC would not be overridden
                    mavc.set_mode('GUIDED') # ensure we're in guided mode to accept velocity commands
                print(f"[TRAJ] Selected traj: {max_traj_idx}/{len(traj_list)-1}", flush=True)
                traj_index = max_traj_idx

        if camera_position is not None:
            camera_position = camera_position[0:-1, -1]
            print("\n[INFO] Current distance to goal (m): ", np.linalg.norm(camera_position-goal_position[0]) if goal_position is not None else "N/A", flush=True)
            print("[INFO] Current RDF coords:", *camera_position, flush=True)
            print("[INFO] Current NED coords:", camera_position[2], camera_position[0], camera_position[1], flush=True)

        # save and view vbg (robust to missing save dir; always attempt visualization)
        try:
            save_dir = os.path.dirname(npz_save_filename) if npz_save_filename else None
            if save_dir and os.path.isdir(save_dir):
                print("\nSaving to {}...".format(npz_save_filename), flush=True)
                try:
                    vbg.vbg.save(npz_save_filename)
                    print("Saving finished", flush=True)
                except Exception as e:
                    print(f"[warning] failed to save VoxelBlockGrid: {e}", flush=True)
            else:
                print("\nSave directory not present or saving disabled; skipping VBG file save\n", flush=True)
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
    except Exception as e:
        shouldStop = True
        print(f"\n[ERROR] Exception in control loop: {e}", flush=True)
        traceback.print_exc()
            
    finally:
        print("Releasing camera capture.")
        stop_keyboard_listener()
        cv2.destroyAllWindows()
        cap.cap.release()

if __name__ == "__main__":
    main()
