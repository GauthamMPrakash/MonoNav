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

realtime_vbg_display = config.get('realtime_vbg_display', False)
realtime_vbg_interval = config.get('realtime_vbg_interval', 5)
realtime_vbg_downsample_voxel_size = config.get('realtime_vbg_downsample_voxel_size', None)

vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device), intrinsic_matrix=fusion_intrinsics)

# Initialize Trajectory Library (Motion Primitives)
trajlib_dir = config['trajlib_dir']
traj_list = get_trajlist(trajlib_dir)
traj_linesets, period, forward_speed, amplitudes = get_traj_linesets(traj_list)
max_traj_idx = int(len(traj_list)/2) # set initial value to that of FORWARD flight (should be median value)
if config['forward_speed'] is not None:
    forward_speed = config['forward_speed']
print(f"\nTrajectory library loaded: {len(traj_list)} trajectories", flush=True)
print("Press 'g' to enable MonoNav autonomous mode, or 'a'/'w'/'d' for manual left/straight/right", flush=True)

# Track printing state for repeated modes so we only print once per mode transition.
last_action_state = None

# Planner: greedy goal-seeking with a simple escape mode to reduce local-minima failures.
planner_cfg = config.get("GreedyEscapePlanner", {}) or {}
planner_params = GreedyEscapePlannerParams(
    enabled=bool(planner_cfg.get("enabled", True)),
    progress_eps_m=float(planner_cfg.get("progress_eps_m", 0.25)),
    stagnation_steps=int(planner_cfg.get("stagnation_steps", 4)),
    escape_min_steps=int(planner_cfg.get("escape_min_steps", 6)),
    unknown_is_unsafe=bool(planner_cfg.get("unknown_is_unsafe", True)),
    known_space_radius_m=float(planner_cfg.get("known_space_radius_m", 0.25)),
)
planner = GreedyEscapePlanner(planner_params)

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

# MAIN MONONAV CONTROL LOOP
def main():
    global shouldStop
    global last_key_pressed
    global max_traj_idx
    global mtx, dist, optimal_mtx, roi
    global goal_position
    global save_dir
    global last_action_state

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
        break
    
    send_espcam_cmd(STREAM_URL[:-10], vflip=1, hmirror=1, res_id=9) # ensure correct orientation and resolution from ESP32-CAM

    vbg_visualizer = None
    vbg_visualizer_pcd = None
    drone_path = None
    drone_marker = None
    drone_positions = []
    if realtime_vbg_display:
        vbg_visualizer, vbg_visualizer_pcd = create_pointcloud_visualizer("MonoNav Realtime VBG", width=960, height=540)

        # Initialize path and drone marker
        drone_path = o3d.geometry.LineSet()
        drone_path.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        drone_path.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        drone_path.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        vbg_visualizer.add_geometry(drone_path)

        drone_marker = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=0.15, cone_height=0.1)
        drone_marker.paint_uniform_color([1.0, 0.0, 0.0])
        drone_marker.compute_vertex_normals()
        vbg_visualizer.add_geometry(drone_marker)

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
        hdg = mavc.get_pose()[5] # get initial yaw
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
        planner.reset()
        last_planner_mode = planner.mode

        start_flight_time = time.perf_counter()
        if FLY_VEHICLE:
            print("Arming Motors!", flush=True)
            mavc.arm()
            print("Taking off.", flush=True)
            mavc.takeoff(height)
            # mavc.set_speed(forward_speed)

        start_pose_thread(20)                      # Start background pose polling at given frequency (non-blocking)

        while True: # Run until VBG is populated to prevent planning error. This is checked by choose_primitive returning without exception
            bgr = cap.read()
            pose = get_latest_pose()[2:]
            camera_position = get_pose_matrix(*pose)
            # COMPUTE DEPTH
            start_time_test = time.perf_counter()
            transform_bgr = transform_image(bgr, mtx, dist, optimal_mtx, roi, enable_undistort)
            transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)
            depth_numpy, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE)
            print("TIME TO COMPUTE DEPTH:", time.perf_counter() - start_time_test)
            cv2.imshow("Frame", bgr)
            cv2.waitKey(1)
            vbg.integration_step(transform_rgb, depth_numpy, camera_position)
            try:
                max_traj_idx = planner.choose(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold,) # Enable DEBUG to print trajectory scores during selection
                break
            except ValueError:
                continue

        print("Starting control.", flush=True)
        print(f"[TRAJ] Estimated next traj: {max_traj_idx}/{len(traj_list)-1}", flush=True)
        traj_counter = 0                    # how many trajectory iterations have we done?
        frame_number = 0                    # Initialize frame counter.
        processing_speed, loop_hz = 0, 0    # for debug display

        while not shouldStop:
            mode = mavc.get_mode()
            if max_traj_idx is None:
                if mode == 'GUIDED' and last_key_pressed == 'g': # Only BRAKE if in Go mode for sudden stop. No need to jerk as much in manual trajectories      
                    mavc.set_mode('BRAKE')
                    time.sleep(0.1) # brief brake before hover to prevent drift during stop
                print("[INFO] No safe trajectory found. Hovering in place.", flush=True)
            else:
                if mode == 'BRAKE':     # so that external mode commands from GCS or RC would not be overridden
                    mavc.set_mode('GUIDED') # ensure we're in guided mode to accept velocity commands
                if last_key_pressed == 'g':
                    traj_str = "Selected traj:"
                else:
                    traj_str = "Estimate next traj:"
                print(f"[TRAJ] {traj_str} {max_traj_idx}/{len(traj_list)-1}", flush=True)
                
            # Check for stop keys first (these exit the control loop). Important that they break out of the loop
            if last_key_pressed == 'p':
                """
                DO NOT USE FOR NORMAL FLYING. WILL STOP MOTORS IMMEDIATELY CAUSING A CRASH. Only use in emergency situations when you need to stop the drone immediately.
                """
                mavc.set_mode('BRAKE')
                mavc.arm(0, force_disarm=True)
                print("Pressed p. EMERGENCY STOP.", flush=True)
                break
            elif last_key_pressed == 'c':
                mavc.set_mode('BRAKE')
                time.sleep(0.2)
                mavc.set_mode('LAND')
                print("Pressed c. Ending control.", flush=True)
                break
            elif last_key_pressed == 'r':
                print("\nPressed r. Switching to SMART_RTL.\n", flush=True)
                if FLY_VEHICLE:
                    mavc.set_mode('SMART_RTL')
                break
            
            elif last_key_pressed == 'f':
                print("Pressed f. Fusing current frame into VBG (no movement).", flush=True)
                traj_index = None

            # Check for trajectory control keys
            elif last_key_pressed == 'a':
                print("Pressed a. Going left.", flush=True)
                traj_index = 0 # left
            elif last_key_pressed == 'w':
                print("Pressed w. Going straight.", flush=True)
                traj_index = len(traj_list)//2 # straight
            elif last_key_pressed == 'd':
                print("Pressed d. Going right.", flush=True)
                traj_index = len(traj_list)-1  # right
            elif last_key_pressed == 'q':
                print("Pressed q. Yawing left.", flush=True)
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=-0.3) # send a yaw left command for 1s
                traj_index = None # no trajectory after the yaw comamnd
                last_key_pressed = None # reset key state immediately after yaw command to prevent key carryover into future periods
            elif last_key_pressed == 'e':
                print("Pressed e. Yawing right.", flush=True)
                if FLY_VEHICLE:
                    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0.3) # send a yaw right command for 1s
                traj_index = None # no trajectory after the backward command
                last_key_pressed = None # reset key state immediately after yaw command to prevent key carryover into future periods
            elif last_key_pressed == 'g':      # GO mode
                if last_action_state != 'g':
                    print("Pressed g. Using MonoNav.", flush=True)
                    last_action_state = 'g'
                traj_index = max_traj_idx
            else:
                mavc.send_body_offset_ned_vel(0, 0)
                print("Hovering in place.", flush=True)
                traj_index = None

            if last_key_pressed != 'g':
                last_key_pressed = None # reset key state if we're not in GO mode, so that single keypresses don't carry over into future periods
            
                if last_action_state not in ('g', 'goal'):
                    last_action_state = None # reset last_action_state if we're not in GO or hover, so that we can print the next mode transition when it happens without being stuck in a state

            start_time = time.perf_counter()
            while time.perf_counter() - start_time < period:
                frame_start_time = time.perf_counter()
                #t0=time.perf_counter()
                bgr = cap.read()
                #t1=time.perf_counter()
                pose = get_latest_pose() # get_latest_pose returns (x, y, z, yaw, pitch, roll) - non-blocking from thread
                #t2=time.perf_counter()

                # Optionally transform camera image (undistort + crop) based on config
                transform_bgr = transform_image(bgr, mtx, dist, optimal_mtx, roi, enable_undistort)
                transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)

                # compute depth
                depth_numpy, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE, make_colormap=True)
                camera_position = get_pose_matrix(*pose[2:]) # we dont need the timestamps
                
                #if last_key_pressed in ('g', 'w', 'a', 'd', 'f'): # if not in hover or land mode, integrate into VBG
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
                        if last_action_state != 'goal':
                            print("Reached goal!")
                        last_action_state = 'goal'
                        break
                
                if save_during_flight:
                    cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg'%(frame_number), bgr)
                    cv2.imwrite(transform_img_dir + '/transform_frame-%06d.rgb.jpg'%(frame_number), transform_bgr)
                    cv2.imwrite(transform_depth_dir + '/' + 'transform_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
                    np.save(transform_depth_dir + '/' + 'transform_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
                    np.savetxt(pose_dir + '/frame-%06d.pose.txt'%(frame_number), camera_position)

                frame_number += 1
            traj_counter += 1

            #print(f"Pose delay (ms): ({(t1-t0)*1000}, Capture delay (ms):{(t2-t1)*1000}", flush=True)
            # Add FPS counter to depth display
            cur_time  = time.perf_counter()
            loop_hz = 1.0 / (cur_time - start_time)
            processing_speed = 1.0 / (cur_time - frame_start_time)

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

                cv2.putText(
                    depth_colormap,
                    fps_text,
                    org=(int((depth_colormap.shape[1] - textsize[0] / 2)), int((textsize[1]) / 2 + hz_size[1] + 5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                    color=(255, 255, 255),
                )
                cv2.imshow("Frame", depth_colormap)
            else:
                cv2.imshow("Frame", transform_bgr)
            cv2.waitKey(1)
   
            # Save trajectory information
            if save_during_flight and max_traj_idx is not None:
                row = np.array([frame_number, int(max_traj_idx), time.perf_counter()-start_flight_time]) # time since start of flight
                with open(save_dir + '/trajectories.csv', 'a') as file:
                    np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')
            
            # Planner always scores trajectories, but only GO mode applies them.
            max_traj_idx = planner.choose(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold,) # Enable DEBUG to print trajectory scores during selection
            if planner.mode != last_planner_mode:
                print(f"[planner] mode={planner.mode}", flush=True)
                last_planner_mode = planner.mode
             
            if shouldStop:        # exit on Esc or if shouldStop
                break
            
            if realtime_vbg_display and traj_counter % realtime_vbg_interval == 0:
                try:
                    pcd = vbg.vbg.extract_point_cloud(weight_threshold)
                    pcd_cpu = pcd.cpu().to_legacy()
                    if realtime_vbg_downsample_voxel_size:
                        pcd_cpu = pcd_cpu.voxel_down_sample(realtime_vbg_downsample_voxel_size)

                    # Update reconstruction points
                    update_pointcloud_visualizer(vbg_visualizer, vbg_visualizer_pcd, pcd_cpu)

                    # Update drone path + marker at current position
                    if camera_position is not None:
                        position_vec = np.array(camera_position[0:3, 3]).reshape(1, 3)
                        drone_positions.append(position_vec[0])

                        # update path
                        pts = np.array(drone_positions)
                        if len(pts) > 1:
                            lines = np.column_stack([np.arange(len(pts)-1), np.arange(1, len(pts))])
                        else:
                            lines = np.zeros((0, 2), dtype=np.int32)
                        drone_path.points = o3d.utility.Vector3dVector(pts)
                        drone_path.lines = o3d.utility.Vector2iVector(lines)
                        drone_path.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]]), (len(lines), 1)))

                        # update arrow pose
                        try:
                            vbg_visualizer.remove_geometry(drone_marker, reset_bounding_box=False)
                        except Exception:
                            pass
                        drone_marker = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.04, cylinder_height=0.15, cone_height=0.1)
                        drone_marker.paint_uniform_color([1.0, 0.0, 0.0])
                        drone_marker.compute_vertex_normals()
                        drone_marker.transform(camera_position)
                        vbg_visualizer.add_geometry(drone_marker)

                    vbg_visualizer.poll_events()
                    vbg_visualizer.update_renderer()
                except Exception as e:
                    print(f"[warning] realtime VBG visualize failed: {e}", flush=True)

        if camera_position is not None:
            camera_position = camera_position[0:-1, -1]
            print("\n[INFO] Current distance to goal (m): ", np.linalg.norm(camera_position-goal_position) if goal_position is not None else "N/A", flush=True)
            print("[INFO] Current RDF coords:", *camera_position, flush=True)
            print("[INFO] Current NED coords:", camera_position[2], camera_position[0], camera_position[1], flush=True)

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
        shouldStop = True
        print("\n[INTERRUPT] Ctrl+C detected. Sent land command.", flush=True)
    except Exception as e:
        shouldStop = True
        print(f"\n[ERROR] Exception in control loop: {e}", flush=True)
        traceback.print_exc()
            
    finally:
        mavc.en_pose_stream(3) # set pose stream to low frequency to reduce bandwidth usage
        print("Releasing camera capture.")
        stop_keyboard_listener()
        if 'cap' in locals() and cap is not None:
            try:
                cap.release()
            except Exception as e:
                print(f"[warning] failed to release camera capture: {e}", flush=True)
        if vbg_visualizer is not None and not realtime_vbg_display:
            try:
                vbg_visualizer.destroy_window()
            except Exception as e:
                print(f"[warning] failed to destroy VBG visualizer window: {e}", flush=True)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
