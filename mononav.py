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
    DStarLitePrimitivePlanner2D,
    motion_primitives_from_traj_list,
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
max_traj_idx = int(len(traj_list)/2)
if config['forward_speed'] is not None:
    forward_speed = config['forward_speed']
print(f"\nTrajectory library loaded: {len(traj_list)} trajectories", flush=True)
print("Press 'g' to enable MonoNav autonomous mode, or 'a'/'w'/'d' for manual left/straight/right", flush=True)

# Track printing state for repeated modes so we only print once per mode transition.
last_action_state = None

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
planner_type = str(planner_cfg.get('type', 'dstar_lite_primitives')).lower()
valid_planner_types = ('dstar_lite_primitives', 'primitive')
if planner_type not in valid_planner_types:
    raise ValueError(f"Unsupported planner.type '{planner_type}'. Expected one of {valid_planner_types}.")
planner_replan_period = planner_cfg.get('replan_period')
if planner_replan_period is None:
    planner_replan_period = float(period)
else:
    planner_replan_period = float(planner_replan_period)
planner_vertical_half_extent = float(planner_cfg.get('vertical_half_extent', max(min_dist2obs, 0.75)))
planner_heading_bins = int(planner_cfg.get('heading_bins', 16))
planner_is_dstar = planner_type == 'dstar_lite_primitives'
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
    try:
        if key.char:
            last_key_pressed = key.char.lower()
    except AttributeError:
        # Handle special keys such as ESC
        if key == keyboard.Key.esc:
            last_key_pressed = 'esc'
        else:
            last_key_pressed = key

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


def _choose_next_primitive(camera_position):
    if camera_position is None:
        return None
    try:
        return choose_primitive(
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
    except ValueError:
        return None


def _poll_opencv_escape():
    key_code = cv2.waitKey(1)
    return key_code != -1 and (key_code & 0xFF) == 27


def _planar_heading_from_pose(camera_pose):
    """
    Return planar heading for the primitive planner's [forward, right] frame.
    """
    if camera_pose is None:
        return 0.0
    forward_axis = np.asarray(camera_pose[:3, 2], dtype=float)
    return float(np.arctan2(forward_axis[0], forward_axis[2]))


# MAIN MONONAV CONTROL LOOP
def main():
    global shouldStop
    global last_key_pressed
    global max_traj_idx
    global mtx, dist, optimal_mtx, roi
    global goal_position
    global save_dir
    global last_action_state
    cap = None
    camera_position = None

    while True:
        # ARDUCOPTER CONTROL
        # Connect to the drone
        print("Connecting to drone...", flush=True)
        mavc.connect_drone(IP, baud=baud)
        print("Connected.")
        mavc.en_pose_stream(25)                    # Commands AP to stream poses at given frequency
        mavc.set_ekf_origin(EKF_LAT, EKF_LON, 0) # Ignored if already close to the previous origin, if set
        reboot = mavc.reboot_if_EKF_origin(0.5)  # Call this function after enabling pose_stream
        if reboot:
            print("Rebooted drone to set EKF origin. Waiting for reconnection...")
            time.sleep(7)    # Wait for drone to reboot
            continue         # Restart connection loop
        mavc.system_time()
        mavc.timesync()      # Probably not required as it is meant to be called in a loop and we don't currently sync pose and frames using timestamps
        break
    
    send_esp_cam_commands(STREAM_URL[0:-10],1,1,9) # send commands to ESP32 cam 9=HVGA, 10=VGA
    # Run the depth model a few times (the first inference is slow), and skip the first few frames
    cap = VideoCapture(STREAM_URL)
    bgr = cap.read()
    frame_height, frame_width = bgr.shape[:2]
    
    # Scale mtx, dist to match current camera resolution
    # Read one frame to get actual resolution
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

    try:
        hdg = mavc.get_pose()[3] # get initial yaw
        # Convert startup nose-relative RDF goal to absolute NED, then reorder
        # to internal [E, D, N] so it matches camera_position[0:-1, -1].
        if goal_position is not None:
            print("\nGoal position (RDF): ", goal_position, flush=True)
            goal_position_ned = rdf_goal_to_ned(goal_position[0], goal_position[1], goal_position[2], hdg)
            mavc.printd(f"Goal position (NED): {[goal_position_ned]}")
            goal_position = np.array([goal_position_ned[1], goal_position_ned[2], goal_position_ned[0]]).reshape(1,3) # In EDN (So it is easier to integrate with O3D whoch uses RDF)
        mavc.printd(f"Heading offset: {hdg*180/np.pi}")
        mavc.set_mode('GUIDED')
        start_keyboard_listener()

        start_keyboard_listener()

        start_flight_time = time.perf_counter()
        if FLY_VEHICLE:
            print("Arming Motors!", flush=True)
            mavc.arm()
            print("Taking off.", flush=True)
            mavc.takeoff(height)
            # mavc.set_speed(forward_speed)
        start_pose_thread(20)                      # Start background pose polling at given frequency (non-blocking)

        # Prime the VBG until primitive selection becomes valid so manual and primitive GO
        # modes start with a meaningful candidate instead of immediately erroring on sparse data.
        while not shouldStop:
            bgr = cap.read()
            pose = get_latest_pose()
            camera_position = get_pose_matrix(*pose)
            transform_bgr = transform_image(bgr, mtx, dist, optimal_mtx, roi, enable_undistort)
            transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)
            depth_numpy, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE)
            vbg.integration_step(transform_rgb, depth_numpy, camera_position)
            max_traj_idx = _choose_next_primitive(camera_position)
            if max_traj_idx is not None:
                break

            if depth_colormap is not None:
                cv2.imshow("Depth", depth_colormap)
            else:
                cv2.imshow("Depth", transform_bgr)
            if _poll_opencv_escape():
                print("Pressed esc. Ending control.", flush=True)
                shouldStop = True
                last_key_pressed = 'c'
                break

        if shouldStop:
            return

        print("Starting control.", flush=True)
        print(f"[TRAJ] Estimated next traj: {max_traj_idx}/{len(traj_list)-1}", flush=True)
        frame_number = 0
        processing_speed, loop_hz = 0, 0
        planned_auto_traj_index = None

        def reset_auto_state():
            nonlocal planned_auto_traj_index
            planned_auto_traj_index = None

        def refresh_dstar_command():
            nonlocal planned_auto_traj_index
            if goal_position is None or camera_position is None:
                reset_auto_state()
                return

            current_position = camera_position[0:-1, -1].astype(float)
            observed_points_xz, obstacle_points_xz = extract_planar_vbg_points(
                vbg.vbg,
                current_down=current_position[1],
                vertical_half_extent=planner_vertical_half_extent,
                filter_weights=filterWeights,
                weight_threshold=weight_threshold,
            )
            mode = mavc.get_mode() if FLY_VEHICLE else 'GUIDED'

            observed_points_fr = observed_points_xz[:, [1, 0]]
            obstacle_points_fr = obstacle_points_xz[:, [1, 0]]
            plan_result = primitive_dstar_planner.plan(
                start_xy=current_position[[2, 0]],
                goal_xy=goal_position[0, [2, 0]],
                obstacle_points_xy=obstacle_points_fr,
                observed_points_xy=observed_points_fr,
                start_heading_rad=_planar_heading_from_pose(camera_position),
            )
            planner_name = "D* Lite primitive"

            if plan_result.found:
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
                return

            planned_auto_traj_index = None
            if mode == 'GUIDED' and FLY_VEHICLE:
                mavc.set_mode('BRAKE')
                time.sleep(0.1)
            print(f"[PLAN] No {planner_name} path: {plan_result.reason}", flush=True)

        while not shouldStop:
            traj_index = None
            fuse_only = False
            mode = mavc.get_mode() if FLY_VEHICLE else 'GUIDED'
            go_mode_active = last_key_pressed == 'g'

            # Refresh planner outputs before selecting the next command period.
            if camera_position is not None:
                max_traj_idx = _choose_next_primitive(camera_position)
                if go_mode_active and planner_is_dstar and goal_position is not None:
                    refresh_dstar_command()

            # Check for stop keys first (these exit the control loop). Important that they break out of the loop
            if last_key_pressed == 'p':
                """
                DO NOT USE FOR NORMAL FLYING. WILL STOP MOTORS IMMEDIATELY CAUSING A CRASH. Only use in emergency situations when you need to stop the drone immediately.
                """
                mavc.set_mode('BRAKE')
                mavc.arm(0, force_disarm=True)
                print("Pressed p. EMERGENCY STOP.", flush=True)
                break
            elif last_key_pressed in ('c', 'esc'):
                mavc.set_mode('BRAKE')
                time.sleep(0.2)
                time.sleep(0.2)
                mavc.set_mode('LAND')
                print("Pressed c. Ending control.", flush=True)
                shouldStop = True
                break
            elif last_key_pressed == 'r':
                print("\nPressed r. Switching to SMART_RTL.\n", flush=True)
                if FLY_VEHICLE:
                    mavc.set_mode('SMART_RTL')
                break
            
            # Check for trajectory control keys
            elif last_key_pressed == 'a':
                print("Pressed a. Going left.", flush=True)
                last_action_state = 'manual'
                reset_auto_state()
                traj_index = 0
                last_key_pressed = None
            elif last_key_pressed == 'w':
                print("Pressed w. Going straight.", flush=True)
                last_action_state = 'manual'
                reset_auto_state()
                traj_index = len(traj_list)//2
                last_key_pressed = None
            elif last_key_pressed == 'd':
                print("Pressed d. Going right.", flush=True)
                last_action_state = 'manual'
                reset_auto_state()
                traj_index = len(traj_list)-1
                last_key_pressed = None
            elif last_key_pressed == 'q':
                print("Pressed q. Yawing left.", flush=True)
                last_action_state = 'yaw'
                reset_auto_state()
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=-0.3)
                last_key_pressed = None
            elif last_key_pressed == 'e':
                print("Pressed e. Yawing right.", flush=True)
                last_action_state = 'yaw'
                reset_auto_state()
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0.3)
                last_key_pressed = None
            elif go_mode_active:
                planner_state = f"go:{planner_type}" if planner_is_dstar and goal_position is not None else "go:primitive"
                if last_action_state != planner_state:
                    reset_auto_state()
                    if planner_is_dstar and goal_position is not None:
                        print("Pressed g. Using MonoNav D* Lite (primitives).", flush=True)
                    else:
                        print("Pressed g. Using MonoNav primitive planner.", flush=True)
                last_action_state = planner_state
                if planner_is_dstar and goal_position is not None:
                    if planned_auto_traj_index is None:
                        refresh_dstar_command()
                if planner_type == 'dstar_lite_primitives' and goal_position is not None:
                    traj_index = planned_auto_traj_index
                else:
                    traj_index = max_traj_idx
            else:
                print("Hovering in place.", flush=True)
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0) # hover in place
                traj_index = None
                last_key_pressed = None
                time.sleep(0.5) # small sleep to prevent busy loop when hovering without a trajectory

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
                    yawrate = -amplitudes[traj_index] * np.sin(np.pi/period*(time.time() - start_time))  # rad/s
                    yvel = yawrate * yvel_gain
                    yawrate = yawrate * yawrate_gain
                    if FLY_VEHICLE:
                        mavc.send_body_offset_ned_vel(forward_speed, yvel, yaw_rate=yawrate)
                
                bgr = cap.read()
                #t1=time.perf_counter()
                pose = get_latest_pose()    # get_latest_pose returns (x, y, z, yaw, pitch, roll) - non-blocking from thread
                #t2=time.perf_counter()

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
                
                if last_key_pressed in ['g', 'w', 'a', 's', 'd']: # if not in hover mode, integrate into VBG
                    vbg.integration_step(transform_rgb, depth_numpy, camera_position)

                if traj_index is not None:
                    yawrate = -amplitudes[traj_index] * np.sin(np.pi/period*(time.perf_counter() - start_time))  # rad/s
                    yvel = yawrate * yvel_gain
                    yawrate = yawrate * yawrate_gain
                    if FLY_VEHICLE:
                        mavc.send_body_offset_ned_vel(forward_speed, yvel, yaw_rate=yawrate)

                if goal_position is not None:
                    dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
                    if dist_to_goal < min_dist2goal:
                        print("Reached goal!")
                        shouldStop = True
                
                if save_during_flight:
                    cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg'%(frame_number), bgr)
                    cv2.imwrite(transform_img_dir + '/transform_frame-%06d.rgb.jpg'%(frame_number), transform_bgr)
                    cv2.imwrite(transform_depth_dir + '/' + 'transform_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
                    np.save(transform_depth_dir + '/' + 'transform_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
                    np.savetxt(pose_dir + '/frame-%06d.pose.txt'%(frame_number), camera_position)

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

            #print(f"Pose delay (ms): ({(t1-t0)*1000}, Capture delay (ms):{(t2-t1)*1000}", flush=True)
            # Add FPS counter to depth display
            cur_time  = time.time()
            loop_hz = 1.0 / (cur_time - start_time)
            processing_speed = 1.0 / (cur_time - frame_start_time)

            # If a key was pressed during the period, let the outer loop consume it next.
            if last_key_pressed in ('p', 'c', 'r', 'a', 'w', 'd', 'h'):
                continue

            # Planner always scores trajectories, but only GO mode applies them.
            max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold,) # Enable DEBUG to print trajectory scores during selection

            mode = mavc.get_mode()
            if max_traj_idx is None:
                if mode == 'GUIDED':      
                    mavc.set_mode('BRAKE')
                    time.sleep(0.1) # brief brake before hover to prevent drift during stop
                print("[INFO] No safe trajectory found. Hovering in place.", flush=True)
            else:
                if mode == 'BRAKE':     # so that external mode commands from GCS or RC would not be overridden
                    mavc.set_mode('GUIDED') # ensure we're in guided mode to accept velocity commands
                if last_key_pressed == 'g':
                    traj_str = "Selected traj:"
                else:
                    traj_str = "Next traj:"
                print(f"[TRAJ] {traj_str} {max_traj_idx}/{len(traj_list)-1}", flush=True)

            # If this cycle ended due to a stop command, do not run planner updates.
            if shouldStop:
                break

        if camera_position is not None:
            camera_position = camera_position[0:-1, -1]
            print("\n[INFO] Current distance to goal (m): ", np.linalg.norm(camera_position-goal_position) if goal_position is not None else "N/A", flush=True)
            print("[INFO] Current RDF coords:", *camera_position, flush=True)
            # print("[INFO] Current NED coords:", camera_position[2], camera_position[0], camera_position[1], flush=True)

        # save and view vbg (robust to missing save dir; always attempt visualization)
        if save_during_flight:
            save_dir = os.path.dirname(npz_save_filename) if npz_save_filename else None
            if save_dir and os.path.isdir(save_dir):
                print("\nSaving to {}...".format(npz_save_filename), flush=True)
                try:
                    vbg.vbg.save(npz_save_filename)
                    print("Saving finished", flush=True)
                except Exception as e:
                    print(f"[warning] failed to save VoxelBlockGrid: {e}", flush=True)
            else:
                print("\nSave directory not present!; skipping VBG file save\n", flush=True)
        else:
            print("\nSaving disabled; skipping VBG file save\n", flush=True)

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
        print("\n[INTERRUPT] Ctrl+C detected. Sent land command.", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Exception in control loop: {e}", flush=True)
        traceback.print_exc()
            
    finally:
        print("Releasing camera capture.")
        stop_pose_thread()
        stop_keyboard_listener()
        if 'cap' in locals() and cap is not None:
            try:
                cap.release()
            except Exception as e:
                print(f"[warning] failed to release camera capture: {e}", flush=True)
        cv2.destroyAllWindows()
        cap.cap.release()

if __name__ == "__main__":
    main()