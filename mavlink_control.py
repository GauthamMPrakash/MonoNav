"""
Library for controlling an ArduCopter vehicle using pymavlink.

Provides functions for connecting, arming, taking off, sending velocity/position commands, changig flight modes and landing.
Also provides functions to receive pose data from the copter.

"""

from pymavlink import mavutil
import numpy as np
import time
import threading

FLTMODES = {"GUIDED": 4, "LOITER":5, "LAND":9, "BRAKE":17}
time_boot, x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0, 0
heading_offset = 0
DEBUG = True                                                # Whether to print debug messages

def printd(string):
    """
    Print debug messages
    """
    if DEBUG:
        print(string)

def send_heartbeat():
    """
    If telemetry is routed through an access point, you may need to send a heartbeat to the autopilot initally to establish conenction using UDP. This is similar in fucntionality to the UDPCl connection in Mission Planner.
    """
    drone.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0)

def connect_drone(IP, baud=115200):
    """ 
    To connect to the drone, use the telemetry module's IP with the mavlink conenction string and port.
    ex: "udpout:192.168.199.51:14550" -> Similar functionality as UDPCl in Misson Planner, but only if a heartbeat is sent from the GCS first if the connection is routed via another Access Point.
    """
    global drone
    drone = mavutil.mavlink_connection(IP, baud)
    printd("Connected")
    send_heartbeat()
    printd("Waiting for heartbeat...")
    drone.wait_heartbeat()
    printd(f"Heartbeat received from system {drone.target_system} component {drone.target_component}")

def set_ekf_origin(lat, lon, alt):
    """
    Set EKF origin for local NED frame
    This is required to use optical flow to get pose estimates without a GPS
    """
    drone.mav.set_gps_global_origin_send(
        0,                  # target_system 0 = broadcast
        int(lat*1e7),       # latitude
        int(lon*1e7),       # longitude
        int(alt*1000)       # altitude in mm
    )
    printd(f"EKF origin set: lat={lat}, lon={lon}, alt={alt}")

def set_mode(mode_name):
    """
    Switch to a flight mode by name (only those defined in FLTMODES)
    """
    if mode_name not in FLTMODES:
        printd(f"Unknown mode: {mode_name}")
        mode_name = "LAND"
    mode_id = FLTMODES[mode_name]
    drone.mav.set_mode_send(
        drone.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    printd(f"Switching to {mode_name}...")
    time.sleep(0.1)

def arm(arm_state=1):
    """
    Arm the drone
    """
    printd("Arming motors...")
    drone.mav.command_long_send(
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        arm_state,0,0,0,0,0,0
    )

    # Wait until armed
    while True:
        msg = drone.recv_match(type='HEARTBEAT', blocking=True)
        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("Vehicle armed")
            break
        time.sleep(0.1)

def takeoff(target_alt):
    """
    Takeoff to target altitude (meters)
    """
    printd(f"Taking off to {target_alt} meters...")
    drone.mav.command_long_send(
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,0,0,0,0,0,
        target_alt
    )

    # Wait until drone reaches target altitude
    while True:
        msg = drone.recv_match(type="VFR_HUD", blocking=True)
        if msg.alt >= target_alt * 0.98:
            printd("Target altitude reached")
            break
        time.sleep(0.1)

def send_body_offset_ned_vel_once(vx, vy, vz=0, yaw_rate=0):
    """
    Send one BODY_NED velocity setpoint packet (non-blocking).
    Useful for high-rate control loops that call this every iteration.
    """
    type_mask = 0b010111000111  # use velocity only
    drone.mav.set_position_target_local_ned_send(
        0,
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        0, 0, 0,   # pos ignored
        vx, vy, vz,
        0, 0, 0,   # acceleration ignored
        0,         # yaw ignored
        yaw_rate
    )

def send_body_offset_ned_vel(vx, vy, vz=0, yaw_rate=0):
    """
    Send velocity in BODY_NED frame (forward/back,left/right,up/down).
    Not absolute NED, but with respect to the drone's current heading.
    """
    printd(f"Sending BODY_NED velocity vx={vx}, vy={vy}, vz={vz}")
    send_body_offset_ned_vel_once(vx, vy, vz=vz, yaw_rate=yaw_rate)

def send_body_offset_ned_pos(x, y, z=0, speed=0, yaw=0, yaw_rate=0):
    """
    Send velocity in BODY_NED frame (forward/back, left/right, up/down).
    The local origin is not the EKF origin, but rather with respect to the current position and heading of the drone.
    """
    type_mask = 0b000111000000
    vx = speed if x > 0 else -speed if x < 0 else 0
    vy = speed if y > 0 else -speed if y < 0 else 0 
    
    printd(f"Sending BODY_NED pos x={x}, y={y}, z={z}")
    drone.mav.set_position_target_local_ned_send(
        0,
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        x,y,z,     # pos
        vx,vy,0,   # velocity
        0,0,0,     # acceleration ignored
        yaw,
        yaw_rate   
    )

def set_speed(speed):
    """
    Set the horizontal navigation speed in meters per second
    """
    printd(f"Setting speed to {speed} m/s")
    drone.mav.command_long_send(
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        1,
        speed,
        -1,
        0,0,0,0  # Unused
    )

def en_pose_stream(freq=15):
    """
    Enable both LOCAL_NED and heading messages to be sent from the autopilot at freqency 'freq'.
    Call once initially to enable the stream. Set frequency to 0 to disable.
    """
    interval = int(1e6 / freq)  # in u_sec

    # LOCAL_POSITION_NED (32)
    drone.mav.command_long_send(
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        32,
        interval,
        0,0,0,0,
        0
    )
    # time.sleep(0.1)

    # ATTITUDE (30)
    drone.mav.command_long_send(
       drone.target_system,
       drone.target_component,
       mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
       0,
       30,
       interval,
       0,0,0,0,
       0
    )

    printd(f"Enabled pose stream at {freq} Hz")

def get_pose(blocking=False):
    """
    Return the pose and attitude of the drone
    """
    global time_boot, x, y, z, roll, pitch, yaw
    while True:
        msg = drone.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=blocking, timeout=0.2)
        
        if not msg or msg.get_type() == "BAD_DATA":
            break

        else:    
            if msg.get_type() == "LOCAL_POSITION_NED":
                time_boot, x, y, z = msg.time_boot_ms, msg.x, msg.y, msg.z
                
            elif msg.get_type() == "ATTITUDE":
                roll, pitch, yaw = msg.roll*180/pi, msg.pitch*180/pi, msg.yaw*180/pi
    return time_boot, x, y, z, yaw, pitch, roll

def heading_offset_init():
    global heading_offset
    """
    Call once to get initial absolute heading.
    Subsequently subtract heading_offset from the absolute heading (ATTITUDE.yaw) to get relative heading
    """
    pose = get_pose()
    heading_offset = pose[4]

def timesync(timeout_s=0.5):
    """
    Send a TIMESYNC request, wait for the matching reply, and return the clock offset.

    Returns:
        offset_ns (int): Estimated AP-local offset in nanoseconds.
                 AP time ~= local_time + offset_ns
                 local_time ~= AP time - offset_ns
        None: If no matching reply is received before timeout.
    """
    t1_ns = time.monotonic_ns()
    drone.mav.timesync_send(
        0,      # tc1=0 indicates request
        t1_ns   # ts1 = local timestamp (ns)
    )

    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None

        msg = drone.recv_match(type='TIMESYNC', blocking=True, timeout=remaining)
        if msg is None:
            return None

        if msg.tc1 == 0:
            continue

        if msg.ts1 != t1_ns:
            continue

        t4_ns = time.monotonic_ns()
        local_mid_ns = (t1_ns + t4_ns) // 2
        offset_ns = msg.tc1 - local_mid_ns
        return offset_ns

def test(): 
    try:
        # Arbitrary location for EKF Origin
        EKF_LAT = 8.4723591
        EKF_LON = 76.9295203
        IP = "udpout:10.208.153.51:14550"  # Drone IP
        connect_drone(IP)
        set_ekf_origin(EKF_LAT, EKF_LON, 0)
        set_mode('GUIDED')
        en_pose_stream()
        # print("Checking telemetry:")
        # for i in range(20):
        #     print(get_pose()[1:])
        #     time.sleep(0.1)
        arm()
        takeoff(0.7)

        set_speed(0.25)
        time.sleep(0.2)
        #send_body_offset_ned_pos_vel(0.7, 0, pos_or_vel="pos", speed=0.3)
        """
        Move along a square

        WARNING: Ensure drone has lots of  space to move. Note that the movement may not move exactly along a square and this will cause the 
                 drone to move at an angel. DO NOT run both functions unless you have tested that the drone will land with an acceptable heading'                 that gives it space for another round.
        """
        yaw_rate = 1
        length = 0.7
        vel = 0.3
        duration = 1
        def square_pos():
            send_body_offset_ned_pos(length,0)
            time.sleep(5)
            send_body_offset_ned_pos(0,length)
            time.sleep(5)
            send_body_offset_ned_pos(-length,0)
            time.sleep(5)
            send_body_offset_ned_pos(0,-length)
            time.sleep(5)
        def square_vel():
            send_body_offset_ned_vel(0.5, 0, yaw_rate=yaw_rate)
            time.sleep(2)
            send_body_offset_ned_vel(0, 0.5, yaw_rate=yaw_rate)
            time.sleep(2)
            send_body_offset_ned_vel(-0.5, 0, yaw_rate=yaw_rate)
            time.sleep(2)
            send_body_offset_ned_vel(0, -0.5, yaw_rate=yaw_rate)
            time.sleep(2)
        #square_vel()
        send_body_offset_ned_pos(length, 0, 0, yaw_rate=0)
        time.sleep(2)
        send_body_offset_ned_pos(0, 0, 0, yaw=90, yaw_rate=1)
        time.sleep(2)
        send_body_offset_ned_pos(length, 0, 0, yaw_rate=0)
        time.sleep(2)
        print("Landing...")
        set_mode('LAND')
        
    except:
        print("Emergency")
        set_mode('LAND')
        arm(0) 
 
if __name__ == '__main__':
    test()
