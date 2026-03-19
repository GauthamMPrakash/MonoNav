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
from utils.dstar_lite import (
    DStarLitePlanner2D,
    DStarLitePrimitivePlanner2D,
    motion_primitives_from_traj_list,
    select_lookahead_waypoint,
)

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
MAX_DEPTH = config['MODEL_MAX_DEPTH'] # maximum depth (in meters) that the depth model can predict
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
if camera_calibration_path:
    mtx, dist, optimal_mtx, roi = get_calibration_values(camera_calibration_path) # for the robot's camera
    calib_width, calib_height = get_calibration_resolution(camera_calibration_path)
    # only compute cropped intrinsics if roi is valid
    if roi is not None:
        fusion_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
    else:
        fusion_intrinsics = None
else:
    # no calibration available; fall back to an ideal matrix once frame size is known
    mtx = dist = optimal_mtx = roi = None
    calib_width = calib_height = None
    fusion_intrinsics = None

# Initialize VoxelBlockGrid
depth_scale = config['VoxelBlockGrid']['depth_scale']
depth_max = config['VoxelBlockGrid']['depth_max']
if depth_max > MAX_DEPTH:
    depth_max = MAX_DEPTH
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

planner_cfg = config.get('planner', {})
planner_type = str(planner_cfg.get('type', 'dstar_lite')).lower()
planner_replan_period = planner_cfg.get('replan_period')
if planner_replan_period is None:
    planner_replan_period = float(period)
else:
    planner_replan_period = float(planner_replan_period)
planner_vertical_half_extent = float(planner_cfg.get('vertical_half_extent', max(min_dist2obs, 0.75)))
planner_waypoint_lookahead = float(planner_cfg.get('lookahead_distance', 0.75))
planner_waypoint_reached_radius = float(planner_cfg.get('waypoint_reached_radius', max(0.35, planner_waypoint_lookahead * 0.5)))
planner_hold_goal_altitude = bool(planner_cfg.get('hold_goal_altitude', True))
planner_heading_bins = int(planner_cfg.get('heading_bins', 16))
planner_is_dstar = planner_type in ('dstar_lite', 'dstar_lite_primitives')

dstar_planner = DStarLitePlanner2D(
    resolution=float(planner_cfg.get('grid_resolution', 0.25)),
    obstacle_buffer_m=float(planner_cfg.get('inflation_radius', min_dist2obs)),
    bounds_padding_m=float(planner_cfg.get('bounds_margin', 1.5)),
    min_window_size_m=float(planner_cfg.get('min_window_size', 8.0)),
    unknown_travel_cost=float(planner_cfg.get('unknown_traversal_cost', 1.75)),
)
primitive_dstar_planner = DStarLitePrimitivePlanner2D(
    motion_primitives=motion_primitives_from_traj_list(traj_list),
    heading_bins=planner_heading_bins,
    resolution=float(planner_cfg.get('grid_resolution', 0.25)),
    obstacle_buffer_m=float(planner_cfg.get('inflation_radius', min_dist2obs)),
    bounds_padding_m=float(planner_cfg.get('bounds_margin', 1.5)),
    min_window_size_m=float(planner_cfg.get('min_window_size', 8.0)),
    unknown_travel_cost=float(planner_cfg.get('unknown_traversal_cost', 1.75)),
)

if planner_is_dstar and goal_position is None:
    print(f"[warning] planner.type is {planner_type} but no goal_position_rdf is configured. GO mode will fall back to primitive selection.", flush=True)
print(f"Planner: {planner_type}", flush=True)
print("Press 'g' for autonomous mode, 'h' to hover, or 'a'/'w'/'d' for manual left/straight/right", flush=True)

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
        last_key_pressed = key


def _planar_heading_from_pose(camera_pose):
    forward_xz = np.asarray(camera_pose[[0, 2], 2], dtype=float)
    norm = float(np.linalg.norm(forward_xz))
    if norm < 1e-6:
        return 0.0
    forward_xz /= norm
    return float(np.arctan2(forward_xz[1], forward_xz[0]))


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


def opencv_waitkey():
    """
    Process OpenCV GUI events; optionally latch Esc into global key state.
    """
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
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

    # Use calibrated intrinsics when available; otherwise fall back to an ideal
    # pinhole matrix sized for the current frame.
    if optimal_mtx is not None and roi is not None:
        vbg_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
    else:
        vbg_intrinsics = get_ideal_intrinsics(frame_width, frame_height)
    vbg.set_intrinsics(vbg_intrinsics)

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
            opencv_waitkey()
        cv2.destroyAllWindows()
        start_keyboard_listener()
        
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
        traj_index = len(traj_list)//2
        auto_mode_active = False
        auto_hold_down = None  # latch altitude in 2D mode
        planned_auto_traj_index = None
        current_waypoint_edn = None
        current_path_xz = np.empty((0, 2), dtype=float)
        camera_position = None

        while not shouldStop:
            traj_index = None

            # Check for stop keys first (these exit the control loop). Important that they break out of the loop
            if last_key_pressed == 'p':
                """
                DO NOT USE FOR NORMAL FLYING. WILL STOP MOTORS IMMEDIATELY CAUSING A CRASH. Only use in emergency situations when you need to stop the drone immediately.
                """
                mavc.set_mode('BRAKE')
                mavc.arm(0)
                print("Pressed p. EMERGENCY STOP.", flush=True)
                shouldStop = True
                break
            elif last_key_pressed == 'c':
                mavc.set_mode('BRAKE')
                mavc.set_mode('LAND')
                print("Pressed c. Ending control.", flush=True)
                shouldStop = True
                break
            elif last_key_pressed == 'r':
                print("\nPressed r. Switching to SMART_RTL.\n", flush=True)
                if FLY_VEHICLE:
                    mavc.set_mode('SMART_RTL')
                shouldStop = True
                break

            elif last_key_pressed == 'h':
                print("Pressed h. Hovering in place.", flush=True)
                auto_mode_active = False
                auto_hold_down = None
                planned_auto_traj_index = None
                current_waypoint_edn = None
                current_path_xz = np.empty((0, 2), dtype=float)
                last_key_pressed = None

            elif last_key_pressed == 'a':
                print("Pressed a. Going left.", flush=True)
                auto_mode_active = False
                auto_hold_down = None
                planned_auto_traj_index = None
                current_waypoint_edn = None
                traj_index = 0
                last_key_pressed = None
            elif last_key_pressed == 'w':
                print("Pressed w. Going straight.", flush=True)
                auto_mode_active = False
                auto_hold_down = None
                planned_auto_traj_index = None
                current_waypoint_edn = None
                traj_index = len(traj_list)//2
                last_key_pressed = None
            elif last_key_pressed == 'd':
                print("Pressed d. Going right.", flush=True)
                auto_mode_active = False
                auto_hold_down = None
                planned_auto_traj_index = None
                current_waypoint_edn = None
                traj_index = len(traj_list)-1
                last_key_pressed = None
            elif last_key_pressed == 'g':
                auto_mode_active = True
                auto_hold_down = None
                planned_auto_traj_index = None
                if planner_is_dstar and goal_position is not None:
                    planner_label = "MonoNav D* Lite" if planner_type == 'dstar_lite' else "MonoNav D* Lite (primitives)"
                    print(f"Pressed g. Using {planner_label}.", flush=True)
                else:
                    print("Pressed g. Using MonoNav primitive planner.", flush=True)
                    traj_index = max_traj_idx
                last_key_pressed = None
            elif auto_mode_active:
                if planner_type == 'dstar_lite' and goal_position is not None:
                    traj_index = None
                elif planner_type == 'dstar_lite_primitives' and goal_position is not None:
                    traj_index = planned_auto_traj_index
                else:
                    traj_index = max_traj_idx
            else:
                print("Hovering in place.", flush=True)
                current_waypoint_edn = None
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0)
                time.sleep(0.1)

            # Save trajectory information
            if save_during_flight:
                traj_idx_to_log = traj_index if traj_index is not None else -1
                row = np.array([frame_number, int(traj_idx_to_log), time.time()-start_flight_time]) # time since start of flight
                with open(save_dir + '/trajectories.csv', 'a') as file:
                    np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')

            command_period = planner_replan_period if auto_mode_active and planner_type == 'dstar_lite' and goal_position is not None else period
            start_time = time.time()
            while time.time() - start_time <= command_period:
                frame_start_time = time.time()
                if traj_index is not None:
                    yawrate = -amplitudes[traj_index] * np.sin(np.pi/period*(time.time() - start_time))  # rad/s
                    yvel = yawrate * yvel_gain
                    yawrate = yawrate * yawrate_gain
                    if FLY_VEHICLE:
                        mavc.send_body_offset_ned_vel(forward_speed, yvel, yaw_rate=yawrate)
                elif auto_mode_active and planner_type == 'dstar_lite' and current_waypoint_edn is not None:
                    if FLY_VEHICLE:
                        mavc.send_local_ned_pos(current_waypoint_edn[2], current_waypoint_edn[0], current_waypoint_edn[1])

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

                if auto_mode_active or traj_index is not None:
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

            # If a key was pressed during the period, let the outer loop consume it next.
            if last_key_pressed in ('p', 'c', 'r', 'a', 'w', 'd', 'g', 'h'):
                continue

            if auto_mode_active and planner_is_dstar and goal_position is not None and camera_position is not None:
                current_position = camera_position[0:-1, -1].astype(float)
                if auto_hold_down is None:
                    auto_hold_down = float(current_position[1])
                observed_points_xz, obstacle_points_xz = extract_planar_vbg_points(
                    vbg.vbg,
                    current_down=current_position[1],
                    vertical_half_extent=planner_vertical_half_extent,
                    filter_weights=filterWeights,
                    weight_threshold=weight_threshold,
                )
                if planner_type == 'dstar_lite_primitives':
                    plan_result = primitive_dstar_planner.plan(
                        start_xy=current_position[[0, 2]],
                        goal_xy=goal_position[0, [0, 2]],
                        obstacle_points_xy=obstacle_points_xz,
                        observed_points_xy=observed_points_xz,
                        start_heading_rad=_planar_heading_from_pose(camera_position),
                    )
                    planner_name = "D* Lite primitive"
                else:
                    plan_result = dstar_planner.plan(
                        start_xy=current_position[[0, 2]],
                        goal_xy=goal_position[0, [0, 2]],
                        obstacle_points_xy=obstacle_points_xz,
                        observed_points_xy=observed_points_xz,
                    )
                    planner_name = "D* Lite"
                mode = mavc.get_mode() if FLY_VEHICLE else 'GUIDED'
                if plan_result.found:
                    current_path_xz = plan_result.world_path
                    if planner_type == 'dstar_lite_primitives':
                        current_waypoint_edn = None
                        planned_auto_traj_index = (
                            int(plan_result.selected_primitive_indices[0])
                            if len(plan_result.selected_primitive_indices) > 0
                            else None
                        )
                        if mode == 'BRAKE' and FLY_VEHICLE:
                            mavc.set_mode('GUIDED')
                        primitive_trace = ""
                        if len(plan_result.selected_primitives) > 0:
                            shown = plan_result.selected_primitives[:6]
                            suffix = "..." if len(plan_result.selected_primitives) > len(shown) else ""
                            primitive_trace = f" prim={shown}{suffix}"
                        print(
                            f"[PLAN] {planner_name} cells={len(plan_result.grid_path)} changed={plan_result.changed_cells} next_traj={planned_auto_traj_index}{primitive_trace}",
                            flush=True,
                        )
                    else:
                        planned_auto_traj_index = None
                        waypoint_xz = select_lookahead_waypoint(
                            plan_result.world_path,
                            current_position[[0, 2]],
                            planner_waypoint_lookahead,
                            planner_waypoint_reached_radius,
                        )
                        target_down = float(goal_position[0, 1]) if planner_hold_goal_altitude else float(auto_hold_down)
                        if waypoint_xz is not None:
                            current_waypoint_edn = np.array([waypoint_xz[0], target_down, waypoint_xz[1]], dtype=float)
                            if mode == 'BRAKE' and FLY_VEHICLE:
                                mavc.set_mode('GUIDED')
                            rdf_waypoint = ned_to_rdf(
                                current_waypoint_edn[2],
                                current_waypoint_edn[0],
                                current_waypoint_edn[1],
                                hdg,
                            )
                            print(
                                f"[PLAN] {planner_name} cells={len(plan_result.grid_path)} changed={plan_result.changed_cells} RDF={np.round(rdf_waypoint, 3)}",
                                flush=True,
                            )
                        else:
                            current_waypoint_edn = None
                            print(f"[PLAN] {planner_name} produced no usable waypoint.", flush=True)
                else:
                    current_path_xz = np.empty((0, 2), dtype=float)
                    current_waypoint_edn = None
                    planned_auto_traj_index = None
                    if mode == 'GUIDED' and FLY_VEHICLE:
                        mavc.set_mode('BRAKE')
                        time.sleep(0.1)
                    print(f"[PLAN] No {planner_name} path: {plan_result.reason}", flush=True)

            elif auto_mode_active and camera_position is not None:
                max_traj_idx = choose_primitive(
                    vbg.vbg,
                    camera_position,
                    traj_linesets,
                    goal_position,
                    min_dist2obs,
                    filterYvals,
                    filterWeights,
                    filterTSDF,
                    weight_threshold,
                )

                mode = mavc.get_mode() if FLY_VEHICLE else 'GUIDED'
                if max_traj_idx is None:
                    if mode == 'GUIDED' and FLY_VEHICLE:
                        mavc.set_mode('BRAKE')
                        time.sleep(0.1)
                    print("[INFO] No safe trajectory found. Hovering in place.", flush=True)
                else: 
                    if mode == 'BRAKE' and FLY_VEHICLE:
                        mavc.set_mode('GUIDED')
                    print(f"[TRAJ] Selected traj: {max_traj_idx}/{len(traj_list)-1}", flush=True)

            # If this cycle ended due to a stop command, do not run planner updates.
            if shouldStop:
                break

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
        stop_pose_thread()
        stop_keyboard_listener()
        cv2.destroyAllWindows()
        if cap is not None:
            cap.cap.release()

if __name__ == "__main__":
    main()
