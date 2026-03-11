"""
Library for controlling an ArduCopter vehicle using pymavlink.

Provides functions for connecting, arming, taking off, sending velocity/position commands, changig flight modes and landing.
Also provides functions to receive pose data from the copter.

"""

from pymavlink import mavutil
import time

FLTMODES = {'GUIDED': 4, 'LOITER':5, 'LAND':9, 'BRAKE':17, 'SmartRTL':21}

DEBUG = True                                                # Whether to print debug messages

def printd(string):
    """
    Print debug messages. Use f strings for multiple variables.
    """
    if DEBUG:
        print(string, flush=True)

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
    printd("Connected")
    send_heartbeat()
    printd("Waiting for heartbeat...")
    drone.wait_heartbeat()
    printd(f"Heartbeat received from system {drone.target_system} component {drone.target_component}")
    return drone

def set_ekf_origin(lat, lon, alt=0):
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
    while arm_state:
        msg = drone.recv_match(type='HEARTBEAT', blocking=True)
        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("Vehicle armed")
            break
        time.sleep(0.1)
    if not arm_state:
        printd("Disarming motors...")

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
        if msg.alt > target_alt * 0.9:
            printd("Target altitude reached")
            break
        time.sleep(0.1)

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
    Return the position (Local NED) and attitude (in radians) of the drone
    Polls for both LOCAL_POSITION_NED and ATTITUDE messages until both are received
    """

    # Keep polling until we get fresh messages (or timeout)
    got_position = False
    got_attitude = False
    
    while True:
        msg = drone.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=blocking, timeout=0.1)
        
        if not msg or msg.get_type() == "BAD_DATA":
            if got_position and got_attitude:
                break
            continue

        if msg.get_type() == "LOCAL_POSITION_NED":
            x, y, z = msg.x, msg.y, msg.z
            got_position = True
        elif msg.get_type() == "ATTITUDE":
            roll, pitch, yaw = msg.roll, msg.pitch, msg.yaw
            got_attitude = True
                
    return x, y, z, yaw, pitch, roll

def heading_offset_init():
    """
    Call once to get initial absolute heading.  Subsequently subtract
    heading_offset from the absolute heading (ATTITUDE.yaw) to get
    relative heading.
    """
    # get_pose returns (x, y, z, yaw, pitch, roll)
    _, _, _, yaw, _, _ = get_pose()
    return yaw

def eSTOP():
    """
    Emergency Motor stop. DISARMS immediately and causes a hard landing. DO NOT USE unless absolutely necessary.
    """
    # set_mode('BRAKE')
    arm(0)

def timesync(timeout_s=0.5):
    """
    Query the autopilot TIMESYNC and return (ap_time_ns, offset_ns).

    - ap_time_ns: estimated current AP clock (nanoseconds)
    - offset_ns: estimated offset between local and AP clocks

    Returns `(None, None)` on timeout/failure.
    """
    t1_ns = time.monotonic_ns()
    drone.mav.timesync_send(0, t1_ns)
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, None
        msg = drone.recv_match(type='TIMESYNC', blocking=True, timeout=remaining)
        if msg is None:
            return None, None
        if msg.tc1 == 0:
            continue
        if msg.ts1 != t1_ns:
            continue
        t4_ns = time.monotonic_ns()
        local_mid_ns = (t1_ns + t4_ns) // 2
        offset_ns = int(msg.tc1 - local_mid_ns)
        ap_ns = int(time.monotonic_ns() + offset_ns)
        return ap_ns
    
def reboot_if_EKF_origin(pos_tolerance=0.2):
    """
    Read the current local‑position and, if either x or y deviates from
    zero by more than `tolerance`, request a reboot so the EKF origin can
    be reset.

    The globals x/y start at 0 and stay there until a LOCAL_POSITION_NED
    message is received.

    Returns True if reboot command was sent, False otherwise.
    """

    for i in range(3):                  # Call multiple times to ensure we get a valid message after enabling the stream
        x, y, _, _, _, _ = get_pose()
        time.sleep(0.05)
    printd(f"reboot check – x={x:.3f}, y={y:.3f}")
    if abs(x) > pos_tolerance or abs(y) > pos_tolerance:
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

def set_param(param_id, param_value, param_type, timeout=5.0):
    """
    UNTESTED

    Set a parameter on the autopilot. Use with caution and ensure you know what the parameter does before changing.
    Waits for PARAM_VALUE response to verify successful parameter set with configurable timeout.
    Returns True if parameter was successfully set, False otherwise.
    """
    printd(f"Setting param {param_id} to {param_value}")
    drone.mav.param_set_send(
        drone.target_system,
        drone.target_component,
        param_id.encode('utf-8'),
        float(param_value),
        param_type
    )
    
    # Wait for PARAM_VALUE response to confirm parameter was set successfully
    start_time = time.time()
    while time.time() - start_time < timeout:
        msg = drone.recv_match(type='PARAM_VALUE', blocking=False, timeout=0.1)
        if msg is not None:
            # Decode param_id from message (may be null-terminated)
            recv_param_id = msg.param_id.split('\x00')[0] if isinstance(msg.param_id, str) else msg.param_id.decode('utf-8').split('\x00')[0]
            
            # Check if this is the parameter we just set
            if recv_param_id == param_id:
                # Verify the value matches (with small tolerance for floating point)
                value_match = abs(msg.param_value - float(param_value)) < 1e-6
                if value_match:
                    printd(f"Parameter {param_id} successfully set to {msg.param_value}")
                    return True
                else:
                    printd(f"ERROR: Parameter {param_id} value mismatch. Expected {param_value}, got {msg.param_value}")
                    return False
    
    printd(f"ERROR: No PARAM_VALUE response received for {param_id} within {timeout}s timeout")
    return False

def test(): 
    """
    Do not fly the vehicle unless you have configured and tested the drone in a safe environment. Always be ready to disarm if the drone behaves unexpectedly.
    """
    try:
        # Arbitrary location for EKF Origin
        EKF_LAT = 8.4723591
        EKF_LON = 76.9295203
        IP = "udpout:10.42.0.110:14550"  # Drone IP
        connect_drone(IP)
        en_pose_stream()
        reboot_if_EKF_origin()
        set_ekf_origin(EKF_LAT, EKF_LON, 0)
        set_mode('GUIDED')
        print("Checking telemetry:")
        for i in range(40):
            pose = get_pose()
            print(pose, flush=True)
            time.sleep(0.1)
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
        length = 0.7
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
        # convert an RDF offset to NED using utility from utils/utils
        from utils import rdf_goal_to_ned
        pos = rdf_goal_to_ned(1, 0.5, 0, get_pose()[3])  # (north,east,down) as tuple
        send_local_ned_pos(pos[0], pos[1], pos[2])
        time.sleep(5)
        print("Landing...")
        set_mode('LAND')
        
    except Exception as e:
        set_mode('LAND')
        print("Emergency")
        print("Exception occurred:", e) 
 
if __name__ == '__main__':
    test()
