"""
Library for controlling an ArduCopter vehicle
"""

from pymavlink import mavutil
from numpy import pi
import time

FLTMODES = {"GUIDED": 4, "LOITER":5, "LAND":9, "BRAKE":17}
DEBUG = True                                               # Whether to print debug messages
time_boot, x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0, 0
heading_offset = 0

def printd(string):
    """
    Print debug messages
    """
    if DEBUG:
        print(string)

def send_heartbeat():
    """
    If telemetry is routed through an access point, you may need to send a heartbeat to the autopilot initally to establish conenction using UDP. This is similar in fucntionality to the UDPCl conenction in Mission Planner.
    """
    drone.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0, 0, 0)

def connect_drone(IP, baud=115200):
    """ 
    To connect to the drone, use the telemtry module's IP with the mavlink conenction string and port.
    ex: "udpout:192.168.199.51:14550" -> Similar functionality as UDPCl in MP, but only if a heartbeat is sent from the GCS first if the connection is routed via another Access Point
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
    print(f"EKF origin set: lat={lat}, lon={lon}, alt={alt}")

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

    while True:
        msg = drone.recv_match(type="VFR_HUD", blocking=True)
        if msg.alt >= target_alt * 0.98:
            printd("Target altitude reached")
            break
        time.sleep(0.1)

def send_body_offset_ned_vel(vx, vy, vz, duration, yaw_rate = 1):
    """
    Send velocity in BODY_NED frame (forward/back,left/right,up/down).
    Not absolute NED, but with respect to the drone's current heading.
    """
    type_mask = 0b010111000111  # use velocity only
    printd(f"Sending BODY_NED velocity vx={vx}, vy={vy}, vz={vz} for {duration}s")
    start = time.time()
    while time.time() - start < duration:
        drone.mav.set_position_target_local_ned_send(
            0,
            drone.target_system,
            drone.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            0,0,0,     # pos ignored
            vx,vy,vz,  # velocity
            0,0,0,     # acceleration ignored
            0,         # yaw ignored
            yaw_rate
        )
        time.sleep(0.1)

def send_body_offset_ned_pos(x, y, z, speed=0.1, yaw_rate = 1):
    """
    Send velocity in BODY_NED frame (forward/back,left/right,up/down).
    The local origin is not the EKF origin, but rather with respect to the current position and heading of the drone.
    """
    type_mask = 0b010111000000
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
        0,         # yaw ignored
        yaw_rate   
    )

def send_body_offset_ned_pos_vel(pos_or_vel, ax, ay, az, speed=0.2, yaw_rate = 1):
    """
    Send position/velocity in BODY_NED frame (forward/back,left/right,up/down).
    The local origin is not the EKF origin, but rather with respect to the current position and heading of the drone.
    """
    type_mask = 0b010111000000
    vx = speed if x > 0 else -speed if x < 0 else 0
    vy = speed if y > 0 else -speed if y < 0 else 0 
    
    if pos_or_vel == "vel" or pos_or_vel == 2:
        tupe_mask = 0b010111000111 # ignore pos
        vx, vy, vz = ax, ay, az

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
        0,         # yaw ignored
        yaw_rate   
    )
    time.sleep(0.1)

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
        speed,
        0,0,0,0,0  # Unused
    )

def land():
    """
    Land the vehicle
    """
    drone.mav.command_long_send(drone.target_system, drone.target_component, mavutil.mavlink.MAV_CMD_NAV_LAND, 0,0,0,0,0,0,0,0)
    printd("Landing...")

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
        msg = drone.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=False)
        
        if not msg or msg.get_type == "BAD_DATA":
            break

        else:    
            if msg.get_type() == "LOCAL_POSITION_NED":
                time_boot, x, y, z = msg.time_boot_ms, msg.x, msg.y, msg.z
                
            elif msg.get_type() == "ATTITUDE":
                roll, pitch, yaw = msg.roll*180/pi, msg.pitch*180/pi, msg.yaw*180/pi
    return time_boot, x, y, z, yaw, pitch, roll

def heading_offset_init():
    pose = get_pose()
    heading = pose[4]

def test(): 
    try:
        # Arbitrary location for EKF Origin
        EKF_LAT = 8.4723591
        EKF_LON = 76.9295203
        IP = "udpout:192.168.199.51:14550"  # Drone IP
        connect_drone(IP)
        set_ekf_origin(EKF_LAT, EKF_LON, 0)
        set_mode("GUIDED")
        arm()
        takeoff(0.7)
        send_body_offset_ned_pos(0.5,0,0,0.1,0)
        time.sleep(2)
        #send_body_offset_ned_pos(0,0.5,0,0.1,0)
        #time.sleep(2)
        send_body_offset_ned_pos(-0.5,0,0,0.1,0)
        time.sleep(2)
        #send_body_offset_ned_pos(0,-0.5,0,0.1,0)
        #time.sleep(2)
        print("Landing...")
        set_mode("LAND")
        
    except:
        print("Emergency")
        land()
        arm(0)

if __name__ == "__main__":
    test()

