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

# For Crazyflie logging
# import logging

# import cflib.crtp
# from cflib.crazyflie import Crazyflie
# from cflib.crazyflie.log import LogConfig
# from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
# from cflib.utils import uri_helper

import mavlink_control as mav          # import the mavlink helper script          
from pynput import keyboard            # Keyboard control

# helper functions
from utils.utils import *

# LOAD VALUES FROM CONFIG FILE
config = load_config('config.yml')

INPUT_SIZE = config['INPUT_SIZE']      # Image size
CHECKPOINT = config['DA2_CHECKPOINT']  # path to checkpoint for DepthAnythingV2
ENCODER = CHECKPOINT[-8:-3]            # extract encoder type from checkpoint filename (assumes format "DA2_{ENCODER}_checkpoint.pth")  
MAX_DEPTH = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DepthAnythingV2 model configurations. You typically only need small or base models
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Initialize the DepthAnythingV2 model and load the checkpoint
depth_anything = DepthAnythingV2(**{**model_configs[config['DA2_ENCODER']], 'max_depth': MAX_DEPTH})
depth_anything.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
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

# URI = uri_helper.uri_from_env(default=config['radio_uri'])
# logging.basicConfig(level=logging.ERROR)
IP = config['IP']
height = config['height']
FLY_VEHICLE = config['FLY_VEHICLE']
baud = config['baud']
EKF_LAT = config['EKF_LAT']
EKF_LON = config['EKF_LON']
SAVE_FRAME_DATA = config.get('SAVE_FRAME_DATA', False)
SAVE_EVERY_N = max(1, int(config.get('SAVE_EVERY_N', 1)))
INTEGRATE_EVERY_N = max(1, int(config.get('INTEGRATE_EVERY_N', 1)))
PRINT_TIMING = config.get('PRINT_TIMING', True)
LOG_EVERY_N_FRAMES = max(1, int(config.get('LOG_EVERY_N_FRAMES', 15)))

# Camera Settings for Undistortion
camera_num = config['camera_num']
# Intrinsics for undistortion
camera_calibration_path = config['camera_calibration_path']
mtx, dist = get_calibration_values(camera_calibration_path) # for the robot's camera
kinect = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault) # for the kinect

# # Initialize Zoedepth Model & Move to device
# conf = get_config('zoedepth', config['zoedepth_mode']) # NOTE: "eval" runs slightly slower, but is stated to be more metrically accurate
# model_zoe = build_model(conf)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Device is: ", DEVICE)
# zoe = model_zoe.to(DEVICE)

STREAM_URL = config['camera_ip']       # YOUR ESP32 HTTP MJPEG stream
#OUTDIR = './esp32_depth'
#os.makedirs(OUTDIR, exist_ok=True)
# Initialize VoxelBlockGrid
depth_scale = config['VoxelBlockGrid']['depth_scale']
depth_max = config['VoxelBlockGrid']['depth_max']
trunc_voxel_multiplier = config['VoxelBlockGrid']['trunc_voxel_multiplier']
weight_threshold = config['weight_threshold'] # for planning and visualization (!! important !!)
device = config['VoxelBlockGrid']['device']
vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device))

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
    goal_position = np.array(config['goal_position']).reshape(1, 3)#np.array([-5., -0.4, 10.0]).reshape(1, 3) # OpenCV frame: +X RIGHT, +Y DOWN, +Z FORWARD
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
kinect_img_dir = os.path.join(save_dir, 'kinect-rgb-images')
kinect_depth_dir = os.path.join(save_dir, 'kinect-depth-images')

os.makedirs(img_dir, exist_ok=True)
os.makedirs(pose_dir, exist_ok=True)
os.makedirs(kinect_img_dir, exist_ok=True)
os.makedirs(kinect_depth_dir, exist_ok=True)

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
# start keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# MAIN MONONAV CONTROL LOOP
def main():
    global shouldStop
    global last_key_pressed
    global max_traj_idx

    # Run the depth model a few times (the first inference is slow), and skip the first few frames
    cap = VideoCapture(STREAM_URL)
    for i in range(0, config['num_pre_depth_frames']):
        bgr = cap.read()
        # COMPUTE DEPTH
        start_time_test = time.time()
        depth_numpy, depth_colormap = compute_depth(depth_anything, bgr, INPUT_SIZE)
        print("TIME TO COMPUTE DEPTH:",time.time()-start_time_test)
        cv2.imshow("frame", bgr)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    # CRAZYFLIE CONTROL
    # Initialize the low-level drivers
    #cflib.crtp.init_drivers()

    # with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    #     cf = scf.cf
    #     reset_estimator(cf)
    #     # Set up log conf
    #     logstate = LogConfig(name='state', period_in_ms=10)
    #     logstate.add_variable('stateEstimate.x', 'float')
    #     logstate.add_variable('stateEstimate.y', 'float')
    #     logstate.add_variable('stateEstimate.z', 'float')
    #     logstate.add_variable('stateEstimate.roll', 'float')
    #     logstate.add_variable('stateEstimate.pitch', 'float')
    #     logstate.add_variable('stateEstimate.yaw', 'float')

    #     # Initialize lists and frame counter.
    #     frame_number = 0

    #     start_flight_time = time.time()
    #     if FLY_CRAZYFLIE:
    #         print("Taking off.")
    #         # Takeoff Sequence
    #         for y in np.linspace(0, height, 21):
    #             cf.commander.send_hover_setpoint(0, 0, 0, y)
    #             time.sleep(0.1)

    #         for _ in range(20):
    #             cf.commander.send_hover_setpoint(0, 0, 0, height)
    #             time.sleep(0.1)

    #     ##########################################
    #     print("Starting control.")
    #     traj_counter = 0 # how many trajectory iterations have we done?
    #     start_time = time.time() # seconds

    #     while not shouldStop:
    #         cv2.imshow('frame', cap.read())
    #         cv2.waitKey(1)
    #         print("shouldStop: ", shouldStop)
    #         if last_key_pressed == 'a':
    #             print("Pressed a. Going left.")
    #             traj_index = 0 # left
    #         elif last_key_pressed == 'w':
    #             print("Pressed w. Going straight.")
    #             traj_index = int(len(traj_list)/2) # straight
    #         elif last_key_pressed == 'd':
    #             print("Pressed d. Going right.")
    #             traj_index = len(traj_list)-1 # right
    #         elif last_key_pressed == 'g':
    #             print("Pressed g. Using MonoNav.")
    #             traj_index = max_traj_idx
    #         elif last_key_pressed == 'c': #end control and land
    #             print("Pressed c. Ending control.")
    #             break
    #         elif last_key_pressed == 'q': #end flight immediately
    #             print("Pressed q. EMERGENCY STOP.")
    #             cf.commander.send_stop_setpoint()
    #             break
    #         else:
    #             print("Else: Staying put.")
    #             start_time = time.time()
    #             while time.time() - start_time < period:
    #                 if FLY_CRAZYFLIE:
    #                     cf.commander.send_hover_setpoint(0, 0, 0, height)
    #                 time.sleep(0.1)
    #             continue
            
    #         # Save trajectory information
    #         row = np.array([frame_number, int(max_traj_idx), time.time()-start_flight_time]) # time since start of flight
    #         with open(save_dir + "/crazyflie_trajectories.csv", "a") as file:
    #             np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')

    #         # Fly the selected trajectory, as applicable.
    #         start_time = time.time()            
    #         while time.time() - start_time < period:
    #             # WARNING: This controller is tuned to work for the Crazyflie 2.1.
    #             # You must check whether your robot follows the open-loop trajectory.
    #             yawrate = -amplitudes[traj_index]*np.sin(np.pi/period*(time.time() - start_time))*180/np.pi # deg/s
    #             yvel =  -yawrate*config['yvel_gain']
    #             yawrate = yawrate*config['yawrate_gain']
    #             print("last_key_pressed = ", last_key_pressed)
    #             if FLY_CRAZYFLIE:
    #                 cf.commander.send_hover_setpoint(forward_speed, yvel, yawrate, height)
    #             # get camera capture and transform intrinsics
    #             crazyflie_rgb = cap.read()
    #             camera_position = get_crazyflie_pose(scf, logstate) # get camera position immediately
    #             if goal_position is not None:
    #                 dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
    #                 print("dist_to_goal: ", dist_to_goal)
    #                 if dist_to_goal < min_dist2goal:
    #                     print("Reached goal!")
    #                     shouldStop = True
    #                     last_key_pressed = 'q'
    #                     break
    #             # Transform Crazyflie Image to Kinect Image
    #             kinect_rgb = transform_image(np.asarray(crazyflie_rgb), mtx, dist, kinect)
    #             kinect_bgr = cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR)
    #             # compute depth
    #             depth_numpy, depth_colormap = compute_depth(kinect_rgb, zoe)

    #             # SAVE DATA TO FILE
    #             cv2.imwrite(crazyflie_img_dir + "/crazyflie_frame-%06d.rgb.jpg"%(frame_number), crazyflie_rgb)
    #             cv2.imwrite(kinect_img_dir + "/kinect_frame-%06d.rgb.jpg"%(frame_number), kinect_rgb)
    #             cv2.imwrite(kinect_depth_dir + '/' + 'kinect_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
    #             np.save(kinect_depth_dir + '/' + 'kinect_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
    #             np.savetxt(crazyflie_pose_dir + "/crazyflie_frame-%06d.pose.txt"%(frame_number), camera_position)
                
    #             # integrate the vbg (prefers bgr)
    #             vbg.integration_step(kinect_bgr, depth_numpy, camera_position)

    #             frame_number += 1
    #         traj_counter += 1

    #         # if crazyflie is not in "GO" (g) mode, reset to stopping mode
    #         if last_key_pressed != 'g':
    #             last_key_pressed = None

    #         shouldStop, max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)
    #         print("SELECTED max_traj_idx: ", max_traj_idx)

    #     # Exited while(!shouldStop); end control!
    #     print("shouldStop: ", shouldStop)
    #     print("Reached goal OR too close to obstacles.")
    #     print("End control.")

    #     if FLY_CRAZYFLIE:
    #         # Stopping sequence
    #         for _ in range(10):
    #                 cf.commander.send_hover_setpoint(0, 0, 0, height)
    #                 time.sleep(0.1)
    #         print("Landing.")
    #         # Landing Sequence
    #         for y in np.linspace(height, 0, 21):
    #             cf.commander.send_hover_setpoint(0, 0, 0, y)
    #             time.sleep(0.1)

    #     cf.commander.send_stop_setpoint()

    #     print("Releasing camera capture.")
    #     cap.cap.release()
    #     cv2.destroyAllWindows()

    #     # save and view vbg
    #     print('Saving to {}...'.format(npz_save_filename))
    #     vbg.vbg.save(npz_save_filename)
    #     print('Saving finished')
    #     print("Visualize raw pointcloud.")
    #     pcd = vbg.vbg.extract_point_cloud(weight_threshold)
    #     o3d.visualization.draw([pcd.cpu()])

        
    # Initialize lists and frame counter.
    frame_number = 0
    start_flight_time = time.time()

    mav.connect_drone(IP, baud=baud)
    mav.set_ekf_origin(EKF_LAT, EKF_LON, 0)
    if FLY_VEHICLE:
        mav.set_mode("GUIDED")
        time.sleep(0.5)
        mav.arm()
        print("Taking off.")
        mav.takeoff(height)
        mav.set_speed(forward_speed)

    ##########################################
    print("Starting control.")
    traj_counter = 0         # how many trajectory iterations have we done?
#   start_time = time.time() # seconds

    while not shouldStop:
        preview_bgr = cap.read()
        cv2.imshow("frame", preview_bgr)
        cv2.waitKey(1)
        if last_key_pressed == 'a':
            traj_index = 0 # left
        elif last_key_pressed == 'w':
            traj_index = int(len(traj_list)/2) # straight
        elif last_key_pressed == 'd':
            traj_index = len(traj_list)-1 # right
        elif last_key_pressed == 'g':
            print("Pressed g. Using MonoNav.")
            traj_index = max_traj_idx
        elif last_key_pressed == 'c': #end control and land
            print("Pressed c. Ending control.")
            mav.set_mode('LAND')
            break
        # elif last_key_pressed == 'q': #end flight immediately
        #     print("Pressed q. EMERGENCY STOP.")
        #     cf.commander.send_stop_setpoint()
        #     break
        else:
            start_time = time.time()
            while time.time() - start_time < period:
                if FLY_VEHICLE:
                    mav.send_body_offset_ned_vel_once(0, 0, 0, yaw_rate=0)
                idle_bgr = cap.read()
                cv2.imshow("frame", idle_bgr)
                cv2.waitKey(1)
                time.sleep(0.1)
            continue

        # Save trajectory information
        row = np.array([frame_number, int(max_traj_idx), time.time()-start_flight_time]) # time since start of flight
        with open(save_dir + '/trajectories.csv', 'a') as file:
            np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')

        # Fly the selected trajectory, as applicable.
        start_time = time.time()
        while time.time() - start_time < period:
            # WARNING: This controller is tuned to work for the Crazyflie 2.1.
            # You must check whether your robot follows the open-loop trajectory.
            yawrate = amplitudes[traj_index]*np.sin(np.pi/period*(time.time() - start_time)) # rad/s
            # yvel = yawrate*config['yvel_gain']
            # yawrate = yawrate*config['yawrate_gain']
            if FLY_VEHICLE:
                #cf.commander.send_hover_setpoint(forward_speed, yvel, yawrate, height)
                mav.send_body_offset_ned_vel_once(forward_speed, 0, 0, yaw_rate=yawrate)
            # get camera capture and transform intrinsics
            # rgb = cap.read()
            frame_start = time.time()
            bgr = cap.read()
            cv2.imshow("frame", bgr)
            cv2.waitKey(1)
            capture_dt = time.time() - frame_start

            pose_start = time.time()
            camera_position = get_drone_pose() # get camera position immediately
            pose_dt = time.time() - pose_start
            if goal_position is not None:
                dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
                if PRINT_TIMING and (frame_number % LOG_EVERY_N_FRAMES == 0):
                    print(f"[goal] dist_to_goal={dist_to_goal:.3f} m")
                if dist_to_goal < min_dist2goal:
                    print("Reached goal!")
                    shouldStop = True
                    last_key_pressed = 'q'
                    break
            # Transform Camera Image to Kinect Image
            # kinect_rgb = transform_image(np.asarray(rgb), mtx, dist, kinect)
            kinect_rgb = None
            # kinect_bgr = cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR)
            # compute depth
            depth_start = time.time()
            depth_numpy, depth_colormap = compute_depth(
                depth_anything,
                bgr,
                INPUT_SIZE,
                make_colormap=(SAVE_FRAME_DATA and (frame_number % SAVE_EVERY_N == 0))
            )
            depth_dt = time.time() - depth_start

            # SAVE DATA TO FILE
            save_dt = 0.0
            if SAVE_FRAME_DATA and (frame_number % SAVE_EVERY_N == 0):
                save_start = time.time()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                kinect_rgb = transform_image(np.asarray(rgb), mtx, dist, kinect)
                cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg'%(frame_number), bgr)
                cv2.imwrite(kinect_img_dir + '/kinect_frame-%06d.rgb.jpg'%(frame_number), kinect_rgb)
                if depth_colormap is not None:
                    cv2.imwrite(kinect_depth_dir + '/' + 'kinect_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
                np.save(kinect_depth_dir + '/' + 'kinect_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
                np.savetxt(pose_dir + '/frame-%06d.pose.txt'%(frame_number), camera_position)
                save_dt = time.time() - save_start
            # integrate the vbg (prefers bgr)
            fuse_dt = 0.0
            if frame_number % INTEGRATE_EVERY_N == 0:
                fuse_start = time.time()
                vbg.integration_step(bgr, depth_numpy, camera_position)
                fuse_dt = time.time() - fuse_start

            if PRINT_TIMING and (frame_number % LOG_EVERY_N_FRAMES == 0):
                total_dt = time.time() - frame_start
                fps = 1.0 / max(total_dt, 1e-6)
                print(
                    f"[timing] cap={capture_dt:.3f}s pose={pose_dt:.3f}s "
                    f"depth={depth_dt:.3f}s fuse={fuse_dt:.3f}s save={save_dt:.3f}s "
                    f"loop={total_dt:.3f}s ({fps:.2f} FPS)"
                )

            frame_number += 1
        traj_counter += 1

        # if crazyflie is not in "GO" (g) mode, reset to stopping mode
        if last_key_pressed != 'g':
            last_key_pressed = None

        shouldStop, max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)
        print("SELECTED max_traj_idx: ", max_traj_idx)

    # Exited while(!shouldStop); end control!
    print("shouldStop: ", shouldStop)
    print("Reached goal OR too close to obstacles.")
    print("End control.")

    if FLY_VEHICLE:
        # Stopping sequence
        print("Landing.")
        mav.land()
        mav.arm(0)

    print("Releasing camera capture.")
    cap.cap.release()
    cv2.destroyAllWindows()

    # save and view vbg
    print("Saving to {}...".format(npz_save_filename))
    vbg.vbg.save(npz_save_filename)
    print("Saving finished")
    print("Visualize raw pointcloud.")
    pcd = vbg.vbg.extract_point_cloud(weight_threshold)
    o3d.visualization.draw([pcd.cpu()])

if __name__ == "__main__":
    main()
