"""
 
"""

import cv2
import torch
import numpy as np
import math as m
import time
import os
import open3d as o3d
import sys
import signal

# Add DepthAnythingV2-metric path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
metric_depth_path = os.path.join(repo_root, 'metric_depth')
sys.path.insert(0, metric_depth_path)

from depth_anything_v2.dpt import DepthAnythingV2         
from pynput import keyboard            # Keyboard control

# helper functions
from utils.utils import *
import mavlink_control as mavc         # import the mavlink helper script 

forward_speed = 0.3

# LOAD VALUES FROM CONFIG FILE
config = load_config('config.yml')

INPUT_SIZE = config['INPUT_SIZE']      # Image size
CHECKPOINT = config['DA2_CHECKPOINT']  # path to checkpoint for DepthAnythingV2
ENCODER = CHECKPOINT[-8:-4]            # extract encoder type from checkpoint filename (assumes format "DA2_{ENCODER}_checkpoint.pth")  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IP = config['IP']                      # dronebridge IP
height = config['height']
FLY_VEHICLE = config['FLY_VEHICLE']
baud = config['baud']
EKF_LAT = config['EKF_LAT']
EKF_LON = config['EKF_LON']
STREAM_URL = config['camera_ip']       # YOUR ESP32 HTTP MJPEG stream

DEPTH_RANGE_M = [0.1, 10.0]            # min and max ranges to be computed
min_depth_cm = int(DEPTH_RANGE_M[0] * 100)  # In cm
max_depth_cm = int(DEPTH_RANGE_M[1] * 100)  # In cm, should be a little conservative
distances_array_length = 72
angle_offset = None
increment_f  = None
distances = np.ones((distances_array_length,), dtype=np.uint16) * (max_depth_cm + 1)

debug_enable = True
display_name  = 'Input/output depth'

# DepthAnythingV2 model configurations. You typically only need small or base models
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    # 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Initialize the DepthAnythingV2 model and load the checkpoint
depth_anything = DepthAnythingV2(**{**model_configs[ENCODER], 'max_depth': DEPTH_RANGE_M[1]})
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
main_loop_should_quit = False

# Camera Settings for Undistortion
camera_num = config['camera_num']
# Intrinsics for undistortion
camera_calibration_path = config['camera_calibration_path']
mtx, dist = get_calibration_values(camera_calibration_path) # for the robot's camera
kinect = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault) # for the kinect

# Initialize VoxelBlockGrid
depth_scale = config['VoxelBlockGrid']['depth_scale']
depth_max = config['VoxelBlockGrid']['depth_max']
trunc_voxel_multiplier = config['VoxelBlockGrid']['trunc_voxel_multiplier']
weight_threshold = config['weight_threshold'] # for planning and visualization (!! important !!)
device = config['VoxelBlockGrid']['device']
vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device))

# # Initialize Trajectory Library (Motion Primitives)
# trajlib_dir = config['trajlib_dir']
# traj_list = get_trajlist(trajlib_dir)
# traj_linesets, period, forward_speed, amplitudes = get_traj_linesets(traj_list)
# max_traj_idx = int(len(traj_list)/2) # set initial value to that of FORWARD flight (should be median value)
# print("Initial trajectory chosen: %d out of %d"%(max_traj_idx, len(traj_list)))

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

# # Save the run information to a csv
# header = ['frame_number', 'chosen_traj_idx', 'time_elapsed']
# with open(save_dir + '/trajectories.csv', 'w') as file:
#     file.write(','.join(header) + '\n')

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

# Use this to rotate all processed data
camera_facing_angle_degree = 0

# Enable/disable each message/function individually
enable_msg_obstacle_distance = True
enable_msg_distance_sensor = False
obstacle_distance_msg_hz_default = 15.0

obstacle_line_height_ratio = 0.18  # [0-1]: 0-Top, 1-Bottom. The height of the horizontal line to find distance to obstacle.
obstacle_line_thickness_pixel = 10 # [1-DEPTH_HEIGHT]: Number of pixel rows to use to generate the obstacle distance message. For each column, the scan will return the minimum value for those pixels centered vertically in the image.

def send_obstacle_distance_message(vehicle):
    global current_time_us, distances, camera_facing_angle_degree
    global last_obstacle_distance_sent_ms
    if current_time_us == last_obstacle_distance_sent_ms:
        # no new frame
        return
    last_obstacle_distance_sent_ms = current_time_us
    if angle_offset is None or increment_f is None:
        mavc.printd("Please call set_obstacle_distance_params before continue")
    else:
        vehicle.mav.obstacle_distance_send(
            current_time_us,    # us Timestamp (UNIX time or time since system boot)
            0,                  # sensor_type, defined here: https://mavlink.io/en/messages/common.html#MAV_DISTANCE_SENSOR
            distances,          # distances,    uint16_t[72],   cm
            0,                  # increment,    uint8_t,        deg
            min_depth_cm,	    # min_distance, uint16_t,       cm
            max_depth_cm,       # max_distance, uint16_t,       cm
            increment_f,	    # increment_f,  float,          deg
            angle_offset,       # angle_offset, float,          deg
            12                  # MAV_FRAME, vehicle-front aligned: https://mavlink.io/en/messages/common.html#MAV_FRAME_BODY_FRD    
        )

# Find the height of the horizontal line to calculate the obstacle distances
#   - Basis: depth camera's vertical FOV, user's input
#   - Compensation: vehicle's current pitch angle
def find_obstacle_line_height():
    global vehicle_pitch_rad, depth_vfov_deg, DEPTH_HEIGHT

    # Basic position
    obstacle_line_height = DEPTH_HEIGHT * obstacle_line_height_ratio

    # Compensate for the vehicle's pitch angle if data is available
    if vehicle_pitch_rad is not None and depth_vfov_deg is not None:
        delta_height = m.sin(vehicle_pitch_rad / 2) / m.sin(m.radians(depth_vfov_deg) / 2) * DEPTH_HEIGHT
        obstacle_line_height += delta_height

    # Sanity check
    if obstacle_line_height < 0:
        obstacle_line_height = 0
    elif obstacle_line_height > DEPTH_HEIGHT:
        obstacle_line_height = DEPTH_HEIGHT
    
    return obstacle_line_height

# Calculate the distances array by dividing the FOV (horizontal) into $distances_array_length rays,
# then pick out the depth value at the pixel corresponding to each ray. Based on the definition of
# the MAVLink messages, the invalid distance value (below MIN/above MAX) will be replaced with MAX+1.
#    
# [0]    [35]   [71]    <- Output: distances[72]
#  |      |      |      <- step = width / 72
#  ---------------      <- horizontal line, or height/2
#  \      |      /
#   \     |     /
#    \    |    /
#     \   |   /
#      \  |  /
#       \ | /           
#       Camera          <- Input: depth_mat, obtained from depth image
#
# Note that we assume the input depth_mat is already processed by at least hole-filling filter.
# Otherwise, the output array might not be stable from frame to frame.
# @njit   # Uncomment to optimize for performance. This uses numba which requires llmvlite (see instruction at the top)
def distances_from_depth_image(obstacle_line_height, depth_mat, distances, min_depth_m, max_depth_m, obstacle_line_thickness_pixel):
    # Parameters for depth image
    depth_img_width  = depth_mat.shape[1]
    depth_img_height = depth_mat.shape[0]

    # Parameters for obstacle distance message
    step = depth_img_width / distances_array_length

    for i in range(distances_array_length):
        # Each range (left to right) is found from a set of rows within a column
        #  [ ] -> ignored
        #  [x] -> center + obstacle_line_thickness_pixel / 2
        #  [x] -> center = obstacle_line_height (moving up and down according to the vehicle's pitch angle)
        #  [x] -> center - obstacle_line_thickness_pixel / 2
        #  [ ] -> ignored
        #   ^ One of [distances_array_length] number of columns, from left to right in the image
        center_pixel = obstacle_line_height
        upper_pixel = center_pixel + obstacle_line_thickness_pixel / 2
        lower_pixel = center_pixel - obstacle_line_thickness_pixel / 2

        # Sanity checks
        if upper_pixel > depth_img_height:
            upper_pixel = depth_img_height
        elif upper_pixel < 1:
            upper_pixel = 1
        if lower_pixel > depth_img_height:
            lower_pixel = depth_img_height - 1
        elif lower_pixel < 0:
            lower_pixel = 0

        # Converting depth from uint16_t unit to metric unit. depth_scale is usually 1mm following ROS convention.
        # dist_m = depth_mat[int(obstacle_line_height), int(i * step)] * depth_scale
        min_point_in_scan = np.min(depth_mat[int(lower_pixel):int(upper_pixel), int(i * step)])
        dist_m = min_point_in_scan * depth_scale

        # Default value, unless overwritten: 
        #   A value of max_distance + 1 (cm) means no obstacle is present. 
        #   A value of UINT16_MAX (65535) for unknown/not used.
        distances[i] = 65535

        # Note that dist_m is in meter, while distances[] is in cm.
        if dist_m > min_depth_m and dist_m < max_depth_m:
            distances[i] = dist_m * 100

# Begin of the main loop

last_time = time.time()
try:
    cap = VideoCapture(STREAM_URL)
    frame = None
    while frame is not None:
        for i in range(3):
            frame = cap.read()
            time.sleep(0.2)
            
    # Connect to the drone
    mavc.connect_drone(IP, baud=baud)
    mavc.set_ekf_origin(EKF_LAT, EKF_LON, 0)

    # Initialize lists and frame counter.
    frame_number = 0
    start_flight_time = time.time()
    mavc.set_mode("GUIDED")
    mavc.arm()
    print("Taking off.")
    mavc.takeoff(height)
    mavc.set_speed(forward_speed)

    DEPTH_HEIGHT, DEPTH_WIDTH = frame.shape[0], frame.shape[1]
    mavc.printd("DEPTH_HEIGHT: {}, DEPTH_WIDTH: {}".format(DEPTH_HEIGHT, DEPTH_WIDTH))

    while not main_loop_should_quit:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        # frames = pipe.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # if not depth_frame:
        #     continue

        # Store the timestamp for MAVLink messages
        current_time_us = int(round(time.time() * 1000000))

        # # Apply the filters
        # filtered_frame = depth_frame
        # for i in range(len(filters)):
        #     if filters[i][0] is True:
        #         filtered_frame = filters[i][2].process(filtered_frame)

        # # Extract depth in matrix form
        # depth_data = filtered_frame.as_frame().get_data()
        # depth_mat = np.asanyarray(depth_data)

        color_frame = cap.read()                            # bgr
        # COMPUTE DEPTH
        start_time_test = time.time()
        depth_mat, depth_frame = compute_depth(depth_anything, color_frame, INPUT_SIZE)

        # Create obstacle distance data from depth image
        obstacle_line_height = find_obstacle_line_height()
        distances_from_depth_image(obstacle_line_height, depth_mat, distances, DEPTH_RANGE_M[0], DEPTH_RANGE_M[1], obstacle_line_thickness_pixel)

        # if RTSP_STREAMING_ENABLE is True:
        #     color_image = np.asanyarray(color_frame.get_data())
        #     rtsp_streaming_img = color_image

        if debug_enable == 1:
            # Prepare the data
            # input_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            # output_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            cv2.imshow("frame", depth_frame)

            # Draw a horizontal line to visualize the obstacles' line
            x1, y1 = int(0), int(obstacle_line_height)
            x2, y2 = int(DEPTH_WIDTH), int(obstacle_line_height)
            line_thickness = obstacle_line_thickness_pixel
            cv2.line(depth_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
            display_image = np.hstack((color_frame, cv2.resize(depth_frame, (DEPTH_WIDTH, DEPTH_HEIGHT))))

            # Put the fps in the corner of the image
            processing_speed = 1 / (time.time() - last_time)
            text = ("%0.2f" % (processing_speed,)) + ' fps'
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(display_image, 
                        text,
                        org = (int((display_image.shape[1] - textsize[0]/2)), int((textsize[1])/2)),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.5,
                        thickness = 1,
                        color = (255, 255, 255))

            # Show the images
            cv2.imshow(display_name, display_image)
            cv2.waitKey(1)

            # Print all the distances in a line
            mavc.printd("%s" % (str(distances)))
            
            last_time = time.time()

except Exception as e:
    mavc.printd(e) 

finally:
    mavc.printd('Closing the script...')
    # start a timer in case stopping everything nicely doesn't work.
    signal.setitimer(signal.ITIMER_REAL, 5)  # seconds...
    if glib_loop is not None:
        glib_loop.quit()
        glib_thread.join()
    pipe.stop()
    mavlink_thread_should_exit = True
    mavlink_thread.join()
    conn.close()
    mavc.printd("INFO: Realsense pipe and vehicle object closed.")
    sys.exit(exit_code)

    
# MAIN MONONAV CONTROL LOOP
def main():
    global main_loop_should_quit
    global last_key_pressed
    # global max_traj_idx

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
        
    # ARDUCOPTER CONTROL
    # Connect to the drone
    mavc.connect_drone(IP, baud=baud)
    mavc.set_ekf_origin(EKF_LAT, EKF_LON, 0)

    # Initialize lists and frame counter.
    frame_number = 0
    start_flight_time = time.time()
    if FLY_VEHICLE==True:
        mavc.set_mode("GUIDED")
        time.sleep(0.1)
        mavc.arm()
        print("Taking off.")
        mavc.takeoff(height)
        mavc.set_speed(forward_speed)

    ##########################################
    print("Starting control.")
    traj_counter = 0         # how many trajectory iterations have we done?
#   start_time = time.time() # seconds

    while not main_loop_should_quit:
        preview_bgr = cap.read()
        cv2.imshow("frame", preview_bgr)
        update_key_from_cv(1)
        if last_key_pressed == 'g':
            print("Pressed g. Using MonoNav.")
            traj_index = max_traj_idx
        elif last_key_pressed == 'c': #end control and land
            mavc.set_mode('LAND')
            print("Pressed c. Ending control.")
            break
        elif last_key_pressed == 'q': #end flight immediately
            mavc.eSTOP()
            print("Pressed q. EMERGENCY STOP.")
            break
        else:
            time.sleep(0.1)
            continue
        
        # # Save trajectory information
        # row = np.array([frame_number, int(max_traj_idx), time.time()-start_flight_time]) # time since start of flight
        # with open(save_dir + '/trajectories.csv', 'a') as file:
        #     np.savetxt(file, row.reshape(1, -1), delimiter=',', fmt='%s')

        # Fly the selected trajectory, as applicable.
        start_time = time.time()            
        while time.time() - start_time < period:
            # WARNING: This controller is tuned for ArduCopter.
            yawrate = amplitudes[traj_index]*np.sin(np.pi/period*(time.time() - start_time)) # rad/s
            yvel = yawrate*config['yvel_gain']
            yawrate = yawrate*config['yawrate_gain']
            if FLY_VEHICLE:
                mavc.send_body_offset_ned_vel(forward_speed, yvel, yaw_rate=yawrate)

            # get camera capture and transform intrinsics
            bgr = cap.read()
            cv2.imshow("frame", bgr)
            update_key_from_cv(1)
            camera_position = get_drone_pose() # get camera position immediately
            if goal_position is not None:
                dist_to_goal = np.linalg.norm(camera_position[0:-1, -1]-goal_position[0])
                if dist_to_goal <= min_dist2goal:
                    print("Reached goal!")
                    main_loop_should_quit = True
                    last_key_pressed = 'c'
                    break
            # Transform Camera Image to Kinect Image
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            kinect_rgb = transform_image(np.asarray(rgb), mtx, dist, kinect)
            kinect_bgr = cv2.cvtColor(kinect_rgb, cv2.COLOR_RGB2BGR)
            # compute depth
            depth_numpy, depth_colormap = compute_depth(depth_anything, bgr, INPUT_SIZE)

            # SAVE DATA TO FILE
            cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg'%(frame_number), bgr)
            cv2.imwrite(kinect_img_dir + '/kinect_frame-%06d.rgb.jpg'%(frame_number), kinect_rgb)
            cv2.imwrite(kinect_depth_dir + '/' + 'kinect_frame-%06d.depth.jpg'%(frame_number), depth_colormap)
            np.save(kinect_depth_dir + '/' + 'kinect_frame-%06d.depth.npy'%(frame_number), depth_numpy) # saved in meters
            np.savetxt(pose_dir + '/frame-%06d.pose.txt'%(frame_number), camera_position)

            # integrate the vbg (prefers bgr)
            vbg.integration_step(bgr, depth_numpy, camera_position)

            frame_number += 1
        traj_counter += 1

        # if not in "GO" (g) mode, reset to stopping mode
        if last_key_pressed != 'g':
            last_key_pressed = None

        main_loop_should_quit, max_traj_idx = choose_primitive(vbg.vbg, camera_position, traj_linesets, goal_position, min_dist2obs, filterYvals, filterWeights, filterTSDF, weight_threshold)
        print("SELECTED max_traj_idx: ", max_traj_idx)

    # Exited while(!main_loop_should_quit); end control!
    print("main_loop_should_quit: ", main_loop_should_quit)
    print("Reached goal OR too close to obstacles.")
    print("End control.")

    if FLY_VEHICLE:
        # Stopping sequence
        print("Landing.")
        mavc.set_mode('LAND')
        mavc.arm(0)

    print("Releasing camera capture.")
    cap.cap.release()
    cv2.destroyAllWindows()

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
        o3d.visualization.draw_geometries([pcd_legacy], window_name="MonoNav Reconstruction")

if __name__ == "__main__":
    main()
