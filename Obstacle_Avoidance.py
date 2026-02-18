import os
import sys
import cv2
from utils.utils import *
import numpy as np
import time
import threading
import torch
from pymavlink import mavutil
from depth_anything_v2.dpt import DepthAnythingV2
from mavlink_control import timesync

# =========================
# Obstacle config
# =========================
distances_array_length = 72
MAX_RANGE_M = 6.0
min_depth_cm = 0
max_depth_cm = int(MAX_RANGE_M * 100)
angle_offset = -30.0
increment_f = 60.0 / distances_array_length
distances = np.ones(distances_array_length, dtype=np.uint16) * 65535

# =========================
# MAVLink globals
# =========================
conn = None
mavlink_thread_should_exit = False
ap_time_offset_ns = 0


# =========================
# MAVLink heartbeat thread
# =========================
def mavlink_loop():
    while not mavlink_thread_should_exit:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            0, 0, 0
        )
        time.sleep(1)


# =========================
# Obstacle computation
# =========================
def compute_obstacles(depth_numpy):
    global distances

    H, W = depth_numpy.shape
    sector_w = W // distances_array_length
    lower_bound = int(0.6 * H)

    for i in range(distances_array_length):
        x_start = i * sector_w
        x_end = (i + 1) * sector_w
        region = depth_m[lower_bound:H, x_start:x_end]

        valid = region[region > 0.1]
        if valid.size == 0:
            distances[i] = 65535
            continue

        d = np.percentile(valid, 10)
        distances[i] = 65535 if d >= MAX_RANGE_M else int(d * 100)


# =========================
# Send OBSTACLE_DISTANCE
# =========================
def send_obstacle_distance(depth_numpy):
    global ap_time_offset_ns

    compute_obstacles(depth_numpy)

    now_ns = time.monotonic_ns()
    ap_time_ns = now_ns + ap_time_offset_ns
    time_usec = int(ap_time_ns / 1000)

    conn.mav.obstacle_distance_send(
        time_usec,
        0,
        distances,
        0,
        min_depth_cm,
        max_depth_cm,
        increment_f,
        angle_offset,
        12
    )


# =========================
# MAIN INITIALIZATION
# =========================
def initialize_mavlink(connection_string, baudrate):
    global conn, ap_time_offset_ns

    print("Connecting to vehicle...")
    conn = mavutil.mavlink_connection(
        connection_string,
        autoreconnect=True,
        source_system=1,
        source_component=93,
        baud=baudrate,
        force_connected=True,
    )

    print("Waiting for heartbeat...")
    conn.wait_heartbeat()

    print("Running TIMESYNC...")
    offset = timesync(conn)

    if offset is not None:
        ap_time_offset_ns = offset
        print(f"Clock synced. Offset ns: {ap_time_offset_ns}")
    else:
        print("TIMESYNC failed. Using local clock.")

    # Start heartbeat thread
    t = threading.Thread(target=mavlink_loop)
    t.daemon = True
    t.start()

    return t

