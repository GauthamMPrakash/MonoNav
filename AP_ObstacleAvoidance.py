"""
MonoNav - Monocular Vision Obstacle Avoidance for ArduCopter
Uses DepthAnythingV2 for metric depth estimation and sends OBSTACLE_DISTANCE
messages to ArduCopter's BendyRuler algorithm for reactive obstacle avoidance.

The companion continuously streams OBSTACLE_DISTANCE to ArduPilot and
periodically sends a GUIDED-mode body-offset position target toward the
goal.  BendyRuler plans a collision-free path while the companion adds
a safety layer (emergency hover when any obstacle < min_obstacle_dist_m).

Required ArduCopter Parameters:
  OA_TYPE          = 1        # BendyRuler path planner
  OA_BR_LOOKAHEAD  = 5        # look-ahead distance (m) — tune for corridor
  OA_MARGIN_MAX    = 2        # max margin from obstacles (m)
  AVOID_ENABLE     = 7        # all avoidance sources
  PRX1_TYPE        = 2        # MAVLink proximity
  PRX1_ORIENT      = 0        # forward
  PRX1_MIN         = 0.2      # m — match DEPTH_RANGE_M[0]
  PRX1_MAX         = 2        # m — match DEPTH_RANGE_M[1]
  WPNAV_SPEED      = 0.5      # m/s — conservative for indoor

Requirements:
- ArduCopter >= v4.5 with BendyRuler enabled
- ESP32-CAM or similar IP camera
- DepthAnythingV2 model checkpoint

"""

import cv2
import torch
import numpy as np
import math as m
import time
import os
import open3d as o3d
import sys
import threading

# Add DepthAnythingV2-metric to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
metric_depth_path = os.path.join(repo_root, 'DepthAnythingV2-metric')
sys.path.insert(0, metric_depth_path)

from depth_anything_v2.dpt import DepthAnythingV2         
from pynput import keyboard                  # Keyboard control

# helper functions
from utils.utils import *
import utils.mavlink_control as mavc         # import the mavlink helper script 

debug_enable = True

# LOAD VALUES FROM CONFIG FILE
config = load_config('config.yml')
forward_speed = config['forward_speed']

INPUT_SIZE = config['INPUT_SIZE']      # Image size
CHECKPOINT = config['DA2_CHECKPOINT']  # path to checkpoint for DepthAnythingV2
ENCODER = CHECKPOINT[-8:-4]            # extract encoder type from checkpoint filename (assumes format "DA2_{ENCODER}_checkpoint.pth")  
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

IP = config['IP']                      # dronebridge IP
height = config['height']
FLY_VEHICLE = config['FLY_VEHICLE']
baud = config['baud']
EKF_LAT = config['EKF_LAT']
EKF_LON = config['EKF_LON']
STREAM_URL = config['camera_ip']       # YOUR ESP32 HTTP MJPEG stream

DEPTH_RANGE_M = [0.2, 20.0]                 # min and max ranges to be computed
min_depth_cm = int(DEPTH_RANGE_M[0] * 100)  # In cm
max_depth_cm = int(DEPTH_RANGE_M[1] * 100)  # In cm, should be a little conservative
distances_array_length = 72
angle_offset = None
increment_f  = None
distances = np.ones((distances_array_length,), dtype=np.uint16) * (max_depth_cm + 1)

# DepthAnythingV2 model configurations. You typically only need small or base models
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#   'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Initialize the DepthAnythingV2 model and load the checkpoint
depth_anything = DepthAnythingV2(**{**model_configs[ENCODER], 'max_depth': DEPTH_RANGE_M[1]})
depth_anything.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
depth_anything = depth_anything.to(DEVICE).eval()
model_device = next(depth_anything.parameters()).device

print(f"[device] torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"[device] selected DEVICE={DEVICE}, model_device={model_device}")
if torch.cuda.is_available():
    print(f"[device] cuda_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
if model_device.type != 'cuda' and torch.cuda.is_available():
    print("[warning] CUDA is available but model is not on CUDA.")

enable_vbg = config['enable_vbg']
save_during_flight = config['save_during_flight']

print(f" VBG (TSDF fusion): {'ENABLED' if enable_vbg else 'DISABLED'}")
print(f"[PERFORMANCE] File saving during flight: {'ENABLED' if save_during_flight else 'DISABLED'}")

# GLOBAL VARIABLES
last_key_pressed = None  # store the last key pressed
shouldStop = False
current_time_us = 0
vehicle_pitch_rad = None
depth_hfov_deg = None
depth_vfov_deg = None
DEPTH_WIDTH = None
DEPTH_HEIGHT = None
ctrl_c_exit = False
distances_lock = threading.Lock()
vehicle_pose_lock = threading.Lock()
obstacle_sender_stop_event = threading.Event()
# Event and thread for periodic timesync calls
timesync_stop_event = threading.Event()

# Intrinsics for undistortion
camera_calibration_path = config['camera_calibration_path']
mtx, dist, optimal_mtx, roi = get_calibration_values(camera_calibration_path) # for the robot's camera
calib_width, calib_height = get_calibration_resolution(camera_calibration_path)
fusion_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)

# Initialize VoxelBlockGrid
depth_scale = config['VoxelBlockGrid']['depth_scale']
obstacle_depth_scale_m_per_unit = 1.0 / float(depth_scale)
depth_max = config['VoxelBlockGrid']['depth_max']
trunc_voxel_multiplier = config['VoxelBlockGrid']['trunc_voxel_multiplier']
weight_threshold = config['weight_threshold'] # for planning and visualization (!! important !!)
# Use cropped intrinsics for VoxelBlockGrid from the start
cropped_mtx = get_cropped_intrinsics(optimal_mtx, roi)
if config['VoxelBlockGrid']['device'] != "None": 
    device = config['VoxelBlockGrid']['device']
else:
    device = 'CUDA:0' if torch.cuda.is_available() else 'CPU:0'
vbg = VoxelBlockGrid(depth_scale, depth_max, trunc_voxel_multiplier, o3d.core.Device(device), intrinsic_matrix=fusion_intrinsics)

# Planning presets
filterYvals = config['filterYvals']
filterWeights = config['filterWeights']
filterTSDF = config['filterTSDF']

# Goal in RDF (right, down, forward) — converted to NED after drone connection
if 'goal_position_rdf' in config:
    goal_position = np.array(config['goal_position_rdf'])
else:
    goal_position = None  # non-directed exploration (manual only)
print("Goal position (RDF):", goal_position)

min_dist2obs = config['min_dist2obs']
min_dist2goal = config['min_dist2goal']
min_obstacle_dist_m = config.get('min_obstacle_dist_m', 0.3)

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
#enable_msg_distance_sensor = False
obstacle_distance_msg_hz_default = 15.0

obstacle_line_height_ratio = 0.4   # [0-1]: 0-Top, 1-Bottom. The height of the horizontal line to find distance to obstacle.
obstacle_line_thickness_pixel = 10 # [1-DEPTH_HEIGHT]: Number of pixel rows to use to generate the obstacle distance message. For each column, the scan will return the minimum value for those pixels centered vertically in the image.

def send_obstacle_distance_message(vehicle):
    global distances, camera_facing_angle_degree
    if angle_offset is None or increment_f is None:
        mavc.printd("Please call set_obstacle_distance_params before continuing")
    else:
        current_time_us = int(round(time.time() * 1000000))
        with distances_lock:
            distances_to_send = distances.copy()

        vehicle.mav.obstacle_distance_send(
            current_time_us,    # us Timestamp (UNIX time or time since system boot)
            0,                  # sensor_type, defined here: https://mavlink.io/en/messages/common.html#MAV_DISTANCE_SENSOR
            distances_to_send,  # distances,    uint16_t[72],   cm
            0,                  # increment,    uint8_t,        deg
            min_depth_cm,	    # min_distance, uint16_t,       cm
            max_depth_cm,       # max_distance, uint16_t,       cm
            increment_f,	    # increment_f,  float,          deg
            angle_offset,       # angle_offset, float,          deg
            12                  # MAV_FRAME, vehicle-front aligned: https://mavlink.io/en/messages/common.html#MAV_FRAME_BODY_FRD    
        )
        
        # Log minimum obstacle distance for monitoring
        # min_obstacle_dist = np.min(distances_to_send)
        # max_obstacle_dist = np.max(distances_to_send)
        # mavc.printd(f"[OBSTACLE] Min, Max distance: {min_obstacle_dist, max_obstacle_dist} cm")

def obstacle_distance_sender_loop(vehicle, send_hz):
    period_s = 1.0 / max(send_hz, 1)
    next_send = time.monotonic()

    while not obstacle_sender_stop_event.is_set():
        send_obstacle_distance_message(vehicle)
        next_send += period_s
        sleep_time = next_send - time.monotonic()
        if sleep_time > 0:
            obstacle_sender_stop_event.wait(sleep_time)
        else:
            next_send = time.monotonic()

# Find the height of the horizontal line to calculate the obstacle distances
#   - Basis: depth camera's vertical FOV, user's input
#   - Compensation: vehicle's current pitch angle
def find_obstacle_line_height():
    global vehicle_pitch_rad, depth_vfov_deg, DEPTH_HEIGHT

    # Basic position
    obstacle_line_height = DEPTH_HEIGHT * obstacle_line_height_ratio

    # Compensate for the vehicle's pitch angle if data is available
    with vehicle_pose_lock:
        pitch_rad = vehicle_pitch_rad
    
    if pitch_rad is not None and depth_vfov_deg is not None:
        delta_height = m.sin(pitch_rad / 2) / m.sin(m.radians(depth_vfov_deg) / 2) * DEPTH_HEIGHT
        obstacle_line_height += delta_height

    # Sanity check
    if obstacle_line_height < 0:
        obstacle_line_height = 0
    elif obstacle_line_height > DEPTH_HEIGHT:
        obstacle_line_height = DEPTH_HEIGHT
    
    return obstacle_line_height

# Set OBSTACLE_DISTANCE parameters using camera intrinsics
def set_obstacle_distance_params_from_intrinsics(camera_matrix, width, height):
    global angle_offset, camera_facing_angle_degree, increment_f
    global depth_hfov_deg, depth_vfov_deg

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    depth_hfov_deg = m.degrees(2 * m.atan(width / (2 * fx)))
    depth_vfov_deg = m.degrees(2 * m.atan(height / (2 * fy)))

    angle_offset = camera_facing_angle_degree - (depth_hfov_deg / 2)
    increment_f = depth_hfov_deg / distances_array_length

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

        # Convert depth units (typically mm) to meters for obstacle checks.
        # VBG depth_scale is typically 1000.0, so meters = value * (1.0 / depth_scale)
        # dist_m = depth_mat[int(obstacle_line_height), int(i * step)] * obstacle_depth_scale_m_per_unit
        min_point_in_scan = np.min(depth_mat[int(lower_pixel):int(upper_pixel), int(i * step)])
        dist_m = min_point_in_scan * obstacle_depth_scale_m_per_unit

        # Default value, unless overwritten: 
        #   A value of max_distance + 1 (cm) means no obstacle is present. 
        #   A value of UINT16_MAX (65535) for unknown/not used.
        distances[i] = 65535

        # Note that dist_m is in meter, while distances[] is in cm.
        if dist_m > min_depth_m and dist_m < max_depth_m:
            distances[i] = dist_m * 100

def _timesync_loop(period_s, stop_event):
    """Background loop that calls mavc.timesync() every `period_s` seconds.
    Exits when `stop_event` is set."""
    next_call = time.monotonic()
    while not stop_event.is_set():
        mavc.timesync()
        next_call += period_s
        sleep_time = next_call - time.monotonic()
        if sleep_time > 0:
            stop_event.wait(sleep_time)

def save_and_visualize_vbg(vbg, npz_save_filename, weight_threshold):
    mavc.printd(" Saving VoxelBlockGrid to {}...".format(npz_save_filename))
    vbg.vbg.save(npz_save_filename)
    mavc.printd(" VoxelBlockGrid saved successfully")

    mavc.printd(" Extracting and visualizing point cloud...")
    pcd = vbg.vbg.extract_point_cloud(weight_threshold)
    pcd_cpu = pcd.cpu()
    pcd_legacy = pcd_cpu.to_legacy()
    num_points = len(pcd_legacy.points)
    mavc.printd("Point cloud has {} points (weight_threshold={})".format(num_points, weight_threshold))
    visualize_pointcloud(pcd_legacy)

# MAIN MONONAV CONTROL LOOP
def main():
    global DEPTH_HEIGHT, DEPTH_WIDTH
    global last_key_pressed
    global shouldStop
    global goal_position
    global mtx, dist, optimal_mtx, roi
    global vehicle_pitch_rad
    global ctrl_c_exit

    goal_nav_active = False  # Track if goal navigation command has been sent

    while True:
        vehicle = mavc.connect_drone(IP, baud=baud)
        mavc.set_ekf_origin(EKF_LAT, EKF_LON)
        mavc.en_pose_stream()
        reboot = mavc.reboot_if_EKF_origin(0.5)
        if reboot:
            mavc.printd("Rebooted drone to set EKF origin. Waiting for reconnection...")
            time.sleep(10)   # Wait for drone to reboot
            mavc.printd("Reconnecting...")
            continue        # Restart connection loop
        break

    sender_thread = None
    last_time = time.time()
    # periodic timesync interval (0.1 Hz)
    timesync_period_s = 10.0  # seconds
    frame_number = 0
    # start timesync thread
    timesync_stop_event.clear()
    timesync_thread = threading.Thread(
        target=_timesync_loop,
        args=(timesync_period_s, timesync_stop_event),
        daemon=True,
    )
    timesync_thread.start()
    
    cap = VideoCapture(STREAM_URL)
    for _ in range(0, max(config['num_pre_depth_frames'], 1)):
        bgr = cap.read()
        compute_depth(bgr, depth_anything, INPUT_SIZE, make_colormap=False)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    # Scale intrinsics if camera resolution differs from calibration resolution
    if bgr is not None:
        if DEPTH_HEIGHT is None or DEPTH_WIDTH is None:
            DEPTH_HEIGHT, DEPTH_WIDTH = bgr.shape[:2]
        mavc.printd("DEPTH_HEIGHT: {}, DEPTH_WIDTH: {}".format(DEPTH_HEIGHT, DEPTH_WIDTH))
        calib_width, calib_height = get_calibration_resolution(camera_calibration_path)
        mtx, dist, optimal_mtx, roi = adjust_intrinsics_to_frame_size(
            mtx, dist, optimal_mtx, roi, DEPTH_WIDTH, DEPTH_HEIGHT, calib_width, calib_height
        )
        if calib_width is not None and calib_height is not None:
            scale_x = DEPTH_WIDTH / calib_width
            scale_y = DEPTH_HEIGHT / calib_height
            if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
                mavc.printd(f"Scaling intrinsics: {calib_width}x{calib_height} → {DEPTH_WIDTH}x{DEPTH_HEIGHT}")
        # Update VoxelBlockGrid with adjusted intrinsics
        vbg_intrinsics = get_cropped_intrinsics(optimal_mtx, roi)
        vbg.intrinsic_matrix = vbg_intrinsics
        vbg.depth_intrinsic = o3d.core.Tensor(vbg_intrinsics, o3d.core.Dtype.Float64)

    else:
        raise RuntimeError("Unable to read first frame from camera")
    
    set_obstacle_distance_params_from_intrinsics(mtx, DEPTH_WIDTH, DEPTH_HEIGHT)

    if enable_msg_obstacle_distance:
        obstacle_sender_stop_event.clear()
        sender_thread = threading.Thread(
            target=obstacle_distance_sender_loop,
            args=(vehicle, obstacle_distance_msg_hz_default),
            daemon=True,
        )
        sender_thread.start()

    if FLY_VEHICLE:
        print("Arming Motors!")
        mavc.set_mode('GUIDED')
        mavc.arm()
        mavc.takeoff(height)
        mavc.set_speed(forward_speed)

    start_pose_thread()  # Start background pose polling at 10 Hz (non-blocking)
    hdg = mavc.heading_offset_init()
    # Convert RDF goal to NED, then reorder to internal [E, D, N]
    # to match camera_position[0:-1, -1] from get_pose_matrix().
    print("Goal position (RDF):", goal_position)
    if goal_position is not None:
        goal_position = np.array(
            rdf_goal_to_ned(goal_position[0], goal_position[1], goal_position[2], hdg),
            dtype=np.float64,
        )
        print(f"Goal position (NED): {goal_position}")
        goal_position = np.array([goal_position[1], goal_position[2], goal_position[0]], dtype=np.float64).reshape(1, 3)
    mavc.printd(f"Heading offset : {mavc.heading_offset*180/np.pi}")

    print("\n=== Keyboard Controls ===")
    if FLY_VEHICLE:
        if goal_position is not None:
            print("  'g' - Start autonomous goal navigation (BendyRuler)")
        else:
            print("  'g' - Move forward at forward_speed m/s (no goal set)")
        print("  'h' - Hover (stop all movement)")
        print("  'c' - Land")
        print("  'q' - EMERGENCY STOP (disarm immediately)")
    print("  'Ctrl-C' - Quit program\n")

    try:
      while not shouldStop:
        bgr = cap.read()
        if debug_enable:
            cv2.imshow("Camera Stream", bgr)
        # Get latest pose directly (no buffering) - read fresh values each iteration
        pos_x, pos_y, pos_z, vehicle_yaw_rad, pitch_rad, vehicle_roll_rad = get_latest_pose()
        with vehicle_pose_lock:
            vehicle_pitch_rad = pitch_rad
        camera_position = get_pose_matrix(pos_x, pos_y, pos_z, vehicle_yaw_rad, pitch_rad, vehicle_roll_rad)

        transform_bgr = transform_image(bgr, mtx, dist, optimal_mtx, roi)
        transform_rgb = cv2.cvtColor(transform_bgr, cv2.COLOR_BGR2RGB)

        depth_mat, depth_colormap = compute_depth(transform_bgr, depth_anything, INPUT_SIZE)

        obstacle_line_height = find_obstacle_line_height()
        with distances_lock:
            distances_from_depth_image(
                obstacle_line_height,
                depth_mat,
                distances,
                DEPTH_RANGE_M[0],
                DEPTH_RANGE_M[1],
                obstacle_line_thickness_pixel,
            )

        # TSDF fusion (optional)
        if enable_vbg:
            vbg.integration_step(transform_rgb, depth_mat, camera_position)
        
        # File writing (optional, for later reruns and data collection - disabled by default for better performance)
        if save_during_flight:
            cv2.imwrite(img_dir + '/frame-%06d.rgb.jpg' % (frame_number,), bgr)
            cv2.imwrite(transform_img_dir + '/transform_frame-%06d.rgb.jpg' % (frame_number,), transform_bgr)
            cv2.imwrite(transform_depth_dir + '/' + 'transform_frame-%06d.depth.jpg' % (frame_number,), depth_colormap)
            np.save(transform_depth_dir + '/' + 'transform_frame-%06d.depth.npy' % (frame_number,), depth_mat)
            np.savetxt(pose_dir + '/frame-%06d.pose.txt' % (frame_number,), camera_position)
        frame_number += 1

        if debug_enable:
            #cv2.imshow("frame", depth_colormap)
            x1, y1 = int(0), int(obstacle_line_height)
            x2, y2 = int(DEPTH_WIDTH), int(obstacle_line_height)
            line_thickness = obstacle_line_thickness_pixel
            cv2.line(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
            # display_image = np.hstack((transform_bgr, depth_colormap))
            display_image = depth_colormap

            processing_speed = 1 / (time.time() - last_time)
            text = ("%0.2f" % (processing_speed,)) + ' fps'
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(
                display_image,
                text,
                org=(int((display_image.shape[1] - textsize[0] / 2)), int((textsize[1]) / 2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(255, 255, 255),
            )
            last_time = time.time()
            cv2.imshow("depth_map", display_image)
            
        update_key_from_cv(1)


        if FLY_VEHICLE:
            if last_key_pressed == 'g':
                if goal_position is not None:
                    # Send position target only once when entering goal navigation mode
                    if not goal_nav_active:
                        print("Pressed g. Using BendyRuler navigation to goal.")
                        mavc.set_mode('GUIDED')
                        mavc.send_local_ned_pos(goal_position[0, 2], goal_position[0, 0], goal_position[0, 1]) # remember, goal_position is now in EDN
                        goal_nav_active = True
                    
                    # Check distance to goal (both in NED frame after heading correction)
                    dist_to_goal = np.linalg.norm(camera_position[0:-1, -1] - goal_position[0])
                    if dist_to_goal <= min_dist2goal:
                        print("Reached goal!")
                        shouldStop = True
                        last_key_pressed = 'c'
                        goal_nav_active = False
                    
                    # # Companion-side safety: emergency hover if obstacle < threshold
                    # with distances_lock:
                    #     valid = distances[distances < (max_depth_cm + 1)]
                    # if valid.size > 0:
                    #     nearest_m = float(np.min(valid)) / 100.0
                    #     if nearest_m < min_obstacle_dist_m:
                    #         mavc.printd(
                    #             "[SAFETY] Obstacle at %.2fm < %.2fm — emergency hover!"
                    #             % (nearest_m, min_obstacle_dist_m)
                    #         )
                    #         mavc.send_body_offset_ned_vel(0, 0, 0, 0)
                else:
                    print("Pressed g. Moving forward.")
                    mavc.send_body_offset_ned_vel(forward_speed, 0, 0, 0)

            elif last_key_pressed == 'h':
                print("Pressed h. Hovering in place.")
                goal_nav_active = False  # Reset flag when switching modes
                last_key_pressed = None
                mavc.send_body_offset_ned_vel(0, 0, 0, 0)
                time.sleep(0.1)
                continue

            elif last_key_pressed == 'c':  # end control and land
                print("Pressed c. Landing.")
                mavc.set_mode('LAND')
                shouldStop = True
                goal_nav_active = False

            elif last_key_pressed == 'q':  # end flight immediately
                print("Pressed q. EMERGENCY STOP.")
                mavc.eSTOP()
                shouldStop = True
                goal_nav_active = False
                

    except KeyboardInterrupt:
        mavc.printd("\n[INTERRUPT] Ctrl+C detected. Quitting...")
        ctrl_c_exit = True
        if FLY_VEHICLE:
            try:
                mavc.printd("[INTERRUPT] Sending immediate LAND command...")
                mavc.set_mode('LAND')
            except Exception:
                pass
        # signal background threads to stop
        obstacle_sender_stop_event.set()
        timesync_stop_event.set()
        shouldStop = True
    except Exception as e:
        mavc.printd(f"\n[ERROR] Exception in main loop: {e}")
        import traceback
        traceback.print_exc()
        shouldStop = True
    finally:
        # Stop background threads and cleanup. If Ctrl-C requested immediate exit,
        # skip long operations (VBG save/visualization) and additional landing.

        mavc.printd(f"[INFO] Current NED coordinates: {camera_position[0:-1, -1]}")    
        mavc.printd("[CLEANUP] Stopping background threads...")
        # Ensure stop events are set
        obstacle_sender_stop_event.set()
        timesync_stop_event.set()
        if sender_thread is not None:
            sender_thread.join(timeout=2.0)
        timesync_thread.join(timeout=2.0)

        # If Ctrl-C was used, we already sent an immediate LAND
        # and we must skip the VBG save/visualization to exit quickly.
        if not ctrl_c_exit:
            # Land vehicle if still flying (only for normal exits via 'c' or 'q')
            if FLY_VEHICLE and shouldStop:
                try:
                    mavc.printd("Landing vehicle...")
                    mavc.set_mode('LAND')
                except Exception:
                    pass

            # Save VoxelBlockGrid (only if enabled)
            if enable_vbg:
                save_and_visualize_vbg(vbg, npz_save_filename, weight_threshold)

        # Release camera and close windows (always do this)
        mavc.printd("[CLEANUP] Releasing camera and closing windows...")
        try:
            cap.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
