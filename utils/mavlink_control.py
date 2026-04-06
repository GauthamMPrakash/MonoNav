"""
Library for controlling an ArduCopter vehicle using pymavlink.

Provides functions for connecting, arming, taking off, sending velocity/position commands, changig flight modes and landing.
Also provides functions to receive pose data from the copter.

"""

from pymavlink import mavutil
import time

FLTMODES = {'GUIDED': 4, 'LOITER':5, 'LAND':9, 'BRAKE':17, 'SMART_RTL':21}
DEBUG = True                                                # Whether to print debug messages

# Initialise poses 
tpos, tatt, x, y, z, yaw, pitch, roll = None, None, None, None, None, None

def printd(string):
    """
    Print debug messages. Use f strings for multiple variables.
    """
    if DEBUG:
        print(f"[mav] {string}", flush=True)

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
    ex: "udpout:192.168.199.51:14550" -> Similar functionality as UDPCl in Mission Planner if you first send out a heartbeat.
    """
    global drone
    drone = mavutil.mavlink_connection(IP, baud, autoreconnect=True)
    printd("Waiting for heartbeat...")
    while not drone.wait_heartbeat(timeout=2):
        send_heartbeat()
    printd(f"Heartbeat received from system {drone.target_system} component {drone.target_component}")
    return drone

def set_ekf_origin(lat=0, lon=0, alt=0):
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
    else:
        mode_id = FLTMODES[mode_name]
        drone.mav.set_mode_send(
            drone.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        printd(f"Switching to {mode_name}...")


def get_mode():
    while True:
        msg = drone.recv_match(type='HEARTBEAT', blocking=False)

        if msg is None:
            continue

        if msg.get_srcComponent() != mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
            continue

        mode_id = msg.custom_mode

        for name, value in FLTMODES.items():
            if value == mode_id:
                return name

        return f"UNKNOWN({mode_id})"

def en_pose_stream(freq=20):
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

def get_pose():
    """
    Return the position (Local NED) and attitude (in radians) of the drone along with the timestamps these messages were polled at.
    """
    got_position = False
    got_attitude = False
    deadline = time.perf_counter() + 0.3  # 300 ms timeout
    while True:
        remaining = deadline - time.perf_counter()
        if remaining < 0:
            printd("Timeout while waiting for pose data")
            return None, None, None, None, None, None, None, None
        msg = drone.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=False, timeout=min(remaining, 0.1))

        if not msg or msg.get_type() == "BAD_DATA":
            continue

        if msg.get_type() == "ATTITUDE":            # mavlink message #30
            tatt, roll, pitch, yaw = msg.time_boot_ms, msg.roll, msg.pitch, msg.yaw
            got_attitude = True
        if msg.get_type() == "LOCAL_POSITION_NED":  # mavlink message #32
            tpos, x, y, z = msg.time_boot_ms, msg.x, msg.y, msg.z
            got_position = True

        if got_position and got_attitude:
            return tpos, tatt, x, y, z, yaw, pitch, roll

def arm(arm_state=1, force_disarm=False):
    """
    Arm the drone
    """
    if arm_state:
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

    if not arm_state:
        if force_disarm:
            force = 21196
        else:
            force = 0
        printd("Disarming motors...")
        drone.mav.command_long_send(
            drone.target_system,
            drone.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            arm_state,force,0,0,0,0,0
        )

def takeoff(target_alt):
    """
    Takeoff to target altitude (meters).

    Blocks until the autopilot reports the vehicle has reached the requested
    altitude (with `target_alt` positive up). The autopilot publishes
    LOCAL_POSITION_NED messages where `z` is Down (positive down), so we
    invert `z` to get altitude above the EKF origin.

    """
    printd(f"Taking off to {target_alt} meters...")
    #target_alt = target_alt + -get_pose()[4]    # add current altitude to get target in EKF frame so that baro offsets are accounted for
    drone.mav.command_long_send(
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,0,0,0,0,0,
        target_alt
    )

    while True:
        # Read LOCAL_POSITION_NED messages only (faster and more robust for altitude)
        alt = -get_pose()[4]    # convert Down (positive) -> altitude (positive up)
        if alt is None:
            time.sleep(0.1)
            continue

        if alt >= target_alt * 0.9:
            printd("Reached target altitude")
            time.sleep(1)  # small delay to ensure we're stable at the target altitude
            return True

def send_body_offset_ned_vel(vx, vy, vz=0, yaw_rate=0):
    """
    Send one BODY_NED velocity setpoint packet (non-blocking).
    Useful for high-rate control loops that call this every iteration.
    """

    #printd(f"Sending BODY_NED vel x={vx}, y={vy}, z={vz}")
    type_mask = 0b010111000111  # use velocity and yaw-rate only
    drone.mav.set_position_target_local_ned_send(
        0,
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        type_mask,
        0, 0, 0,                # pos ignored
        vx, vy, vz,
        0, 0, 0,                # acceleration ignored
        0,                      # yaw ignored
        yaw_rate
    )

def send_local_ned_pos(x, y, z):
    """
    Send position in LOCAL_NED frame (Relative to EKF-origin).
    Currently AP_ObstacleAvoidance only requires 2D position control.
    """
    type_mask = 0b110111111000
    # vx = speed if x > 0 else -speed if x < 0 else 0
    # vy = speed if y > 0 else -speed if y < 0 else 0 
    
    printd(f"Sending LOCAL_NED pos North={x}, East={y}, Down={z}")
    drone.mav.set_position_target_local_ned_send(
        0,
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x,y,z,     # pos
        0,0,0,     # velocity
        0,0,0,     # acceleration ignored
        0,
        0  
    )

def set_speed(speed):
    """
    Set the horizontal navigation [ground] speed in meters per second
    """
    printd(f"Setting speed to {speed} m/s")
    drone.mav.command_long_send(
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,
        1,       # 0 = airspeed, 1 = groundspeed
        speed,
        0,
        0,0,0,0  # Unused
    )

def timesync(timeout_s=0.5):
    """
    Query the autopilot TIMESYNC and return (ap_time_ns, offset_ns).

    - ap_time_ns: estimated current AP clock (nanoseconds)
    - offset_ns: estimated offset between local and AP clocks

    Returns: offset_ns
    """
    t1_ns = time.monotonic_ns()
    drone.mav.timesync_send(0, t1_ns)
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
        offset_ns = int(msg.tc1 - local_mid_ns)
        printd("Offset_ns between GCS and AP = {offset_ns} ms".format(offset_ns=offset_ns/1e6))
        return offset_ns

def system_time():      # mavlink message #2
    """
    Send companion wall-clock time to ArduPilot using SYSTEM_TIME.

    - time_unix_usec: UNIX epoch time in microseconds
    - time_boot_ms: companion monotonic time since boot in milliseconds
    """
    unix_us = time.time_ns() // 1000
    boot_ms = int(time.monotonic_ns() // 1_000_000)
    drone.mav.system_time_send(unix_us, boot_ms)
    printd(f"Sent SYSTEM_TIME unix_us={unix_us}")

def reboot_if_EKF_origin(pos_tolerance=0.3):
    """
    Read the current local‑position and, if either x or y deviates from
    zero by more than `tolerance`, request a reboot so the EKF origin can
    be reset.

    The globals x/y start at 0 and stay there until a LOCAL_POSITION_NED
    message is received.

    Returns True if reboot command was sent, False otherwise.
    """

    _, _, x, y, z, _, _, _ = get_pose()
    printd(f"reboot check – x={x:.3f}, y={y:.3f}")
    if abs(x) > pos_tolerance or abs(y) > pos_tolerance or abs(z) > pos_tolerance:
        printd(f"pos deviation exceeds {pos_tolerance}, rebooting")
        drone.mav.command_long_send(
            drone.target_system,
            drone.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,
            1,
            0, 0, 0,
            0, 0, 0
        )
        return True
    return False

def test():
    """
    Do not fly the vehicle unless you have configured and tested the drone in a safe environment. Always be ready to disarm if the drone behaves unexpectedly.
    """
    try:
        # Arbitrary location for EKF Origin
        EKF_LAT = 8.4723591
        EKF_LON = 76.9295203
        IP = "udpout:192.168.53.51:14550"  # Drone IP
        connect_drone(IP)
        reboot_if_EKF_origin()
        set_ekf_origin(EKF_LAT, EKF_LON, 0)
        set_mode('GUIDED')
        print("Checking telemetry:")
        for i in range(40):
            pose = get_pose()
            print(pose[2:], flush=True)
        print("AP time, offset:", timesync())
        arm()
        takeoff(1)
        set_speed(0.3)
        """
        Move along a square

        WARNING: Ensure drone has lots of  space to move. Note that the movement may not move exactly along a 
        square and this will cause the drone to move at an angel. DO NOT run both functions unless you have 
        tested that the drone will land with an acceptable heading' that gives it space for another round.
        """
        yaw_rate = 1
        vel = 0.5
        def square_vel():
            send_body_offset_ned_vel(vel, 0, yaw_rate=yaw_rate)
            time.sleep(2)
            send_body_offset_ned_vel(0, vel, yaw_rate=yaw_rate)
            time.sleep(2)
            send_body_offset_ned_vel(-vel, 0, yaw_rate=yaw_rate)
            time.sleep(2)
            send_body_offset_ned_vel(0, -vel, yaw_rate=yaw_rate)
            time.sleep(2)
        #square_vel()
        print("Landing...")
        set_mode('LAND')
        
    except Exception as e:
        set_mode('LAND')
        print("Emergency")
        print("Exception occurred:", e) 
    finally:
        set_mode('LAND')
if __name__ == '__main__':
    test()