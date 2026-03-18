"""Fun script to control an ArduCopter using PC keyboard!

Keys:
  w/s  -> forward/backward
  a/d  -> left/right
  q/e  -> yaw left/right
  h/n  -> increase/decrease altitude in 5cm steps
  t    -> takeoff (reads altitude from config.yml; arms drone first)
  l    -> land (sets mode to LAND)

Holding a direction or yaw key will continuously send the velocity command;
when all movement keys are released the controller immediately sends a
zero-velocity packet to hover.

SETTING SAFETY BOUNDS:
----------------------
The script also enforces configurable safety bounds on the drone's position by setting
bl (left bound), br (right bound), bf (forward bound), and bb (backward bound).
The bounds are defined in the `local body offset frame` at the drone's initial position and based on 
its heading on the ground at startup.
-------------------------------------
Set any of the bounds to None to disable that bound, or set them to a positive value in meters.

Run this script independently when you want to manually fly the drone.
"""

"""
ALWAYS KEEP IN MIND THE POSSIBILITY OF FAILURE TO RESPECT THE BOUNDS AS MAY BE THE CASE IF THE SENSORS 
ARE BAD OR EKF FAILS. SO ALWAYS FLY WITH CAUTION!
"""
bl, br, bf, bb = -0.3, 0.3, 10, -1
# The example values above create a safety bound with 20 cm to either side, no forward bound, and a backward bound of 50 cm. Adjust as needed for your environment and testing purposes.

import os
import sys
import time
from pynput import keyboard
from pynput.keyboard import Key

# Ensure the repository root is on sys.path so we can import `utils` from anywhere
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from pymavlink import mavutil
from utils.utils import load_config, start_pose_thread, get_latest_pose, stop_pose_thread
import utils.mavlink_control as mavc

# ---------------------------------------------------------------------------
# configuration constants
# ---------------------------------------------------------------------------
CONFIG_FILE = '../config.yml'
# single speed parameter used for both forward/backward and strafe
SPEED = 0.5              # horizontal speed (m/s); adjustable with arrow keys
YAW_RATE = 70.0          # deg/s when pressing 'q' or 'e'; adjustable with left/right arrows
COMMAND_HZ = 15          # how many velocity commands per second
ALT_SPEED = 0.25         # m/s vertical speed when pressing 'h' or 'n'
# speed bounds
MIN_SPEED = 0.1
MAX_SPEED = 1.0
# yaw rate bounds (degrees per second)
MIN_YAW_RATE = 10.0
MAX_YAW_RATE = 100.0
# (altitude control is handled as a velocity term; no helper needed)

# global state updated by keyboard callbacks
_pressed_keys = set()
# track last press time to avoid stuck keys if release event is missed
_key_timestamps = {}
# takeoff/land flags set by keyboard, handled in main loop
_takeoff_requested = False
_land_requested = False

# helper functions for keyboard events

def _on_press(key):
    """Called by ``pynput`` when a key is pressed."""
    # handle non-character keys first (arrows, etc.)
    if key == Key.up:
        global SPEED
        SPEED = min(MAX_SPEED, SPEED + 0.1)
        print(f"[KB] speed increased to {SPEED:.2f} m/s")
        return
    if key == Key.down:
        SPEED = max(MIN_SPEED, SPEED - 0.1)
        print(f"[KB] speed decreased to {SPEED:.2f} m/s")
        return
    if key == Key.left:
        global YAW_RATE
        YAW_RATE = max(MIN_YAW_RATE, YAW_RATE - 10.0)
        print(f"[KB] yaw rate decreased to {YAW_RATE:.1f} deg/s")
        return
    if key == Key.right:
        YAW_RATE = min(MAX_YAW_RATE, YAW_RATE + 10.0)
        print(f"[KB] yaw rate increased to {YAW_RATE:.1f} deg/s")
        return

    try:
        k = key.char
    except AttributeError:
        return

    # takeoff/land are one-shot actions (do not stay in the pressed set)
    if k == 't':
        global _takeoff_requested
        print("[KB] takeoff requested (will execute in main loop)")
        _takeoff_requested = True
        return
    if k == 'l':
        global _land_requested
        print("[KB] land requested (will execute in main loop)")
        _land_requested = True
        return

    # add movement keys to the set so the main loop knows which directions
    # are currently held down.
    if k in ('w', 'a', 's', 'd', 'q', 'e', 'h', 'n'):
        # record press and timestamp; height keys handled separately
        _pressed_keys.add(k)
        _key_timestamps[k] = time.monotonic()
    # allow explicit quit key
    if k == 'x':
        print("[KB] exit requested")
        raise KeyboardInterrupt


def _on_release(key):
    """Called by ``pynput`` when a key is released."""
    try:
        k = key.char
    except AttributeError:
        return
    if k in _pressed_keys:
        _pressed_keys.discard(k)
        _key_timestamps.pop(k, None)
    # allow Ctrl-C to stop listener
    if k == '\x03':
        return False


# ---------------------------------------------------------------------------
# main control loop
# ---------------------------------------------------------------------------

def main():
    global config
    global bl, br, bf, bb

    if bl >= br or bb >= bf:
        print("[KB] WARNING: invalid safety bounds! No movement will be allowed.")
        print("[KB] Please ensure bl < br and bb < bf, or set bounds to None to disable.")
        return
    
    config = load_config(CONFIG_FILE)

    EKF_LAT = config.get('EKF_LAT')
    EKF_LON = config.get('EKF_LON')

    # connect and prepare the vehicle
    print(f"[KB] connecting to drone at {config.get('IP')} (baud {config.get('baud',115200)})")
    mavc.connect_drone(config['IP'], baud=config.get('baud', 115200))
    mavc.set_ekf_origin(EKF_LAT, EKF_LON)
    mavc.reboot_if_EKF_origin()  # Reset EKF origin to ensure the safety bounds work as intended
    mavc.en_pose_stream(15)
    init_heading = mavc.heading_offset_init()
    print(f"[KB] initial heading offset: {np.rad2deg(init_heading):.1f} deg")
    print("[KB] vehicle should now be in GUIDED mode")

    # turn off local echo so keystrokes aren't printed
    try:
        import termios, sys
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ECHO  # lflags
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
    except Exception:
        old_settings = None
    # start listener (don't suppress; Ctrl-C will still work)
    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    print("[KB] Keyboard teleop started.  't' takeoff, 'l' land, Ctrl-C to exit.")

    try:
        global _takeoff_requested, _land_requested
        period = 1.0 / COMMAND_HZ
        takeoff_in_progress = False
        while True:
            # clear any stale keys that haven't been updated in >1s
            now = time.monotonic()
            for k, t in list(_key_timestamps.items()):
                if now - t > 1.0:
                    _pressed_keys.discard(k)
                    _key_timestamps.pop(k, None)

            # debug print of state
            # print(f"[KB] flags: takeoff={_takeoff_requested}, inprog={takeoff_in_progress}")
            # Handle takeoff/land requests from keyboard callbacks
            # (execute in main thread, not listener thread, to avoid blocking)
            if _takeoff_requested and not takeoff_in_progress:
                alt = 0.7
                print(f"[KB] arming and taking off to {alt} m")
                mavc.set_mode('GUIDED')
                mavc.arm()
                print("[KB] armed, now taking off...")
                mavc.takeoff(alt)
                _takeoff_requested = False
                takeoff_in_progress = True
                start_pose_thread(10)
            
            if _land_requested:
                print("[KB] switching to LAND mode")
                mavc.set_mode('LAND')
                time.sleep(3)
                mavc.arm(0)
                _land_requested = False
                takeoff_in_progress = False
                _takeoff_requested = False  # drop any lingering request
                # give the autopilot a moment to settle on the ground
                time.sleep(1)
                print("[KB] landed and disarmed, ready for next takeoff")
                # stop pose thread so that the pose mavlink stream doesn't overwhelm the system casuing arming check to fail
                stop_pose_thread()

            vx = 0.0
            vy = 0.0
            yaw_rate = 0.0
            vz = 0.0

            # linear velocity (using unified SPEED)
            pose = get_latest_pose()
            x_n, y_e = pose[0], pose[1]
            c, s = np.cos(init_heading), np.sin(init_heading)
            # rotate NED -> local frame where +x is "forward" at init_heading
            x_fwd = c * x_n + s * y_e
            y_right = -s * x_n + c * y_e

            if 'w' in _pressed_keys:
                if x_fwd and x_fwd <= bf:
                    vx += SPEED
                else:
                    print("[KB] forward bound reached")
                    time.sleep(0.1)
            if 's' in _pressed_keys:
                if x_fwd and x_fwd >= bb:
                    vx -= SPEED
                else:
                    print("[KB] backward bound reached")
                    time.sleep(0.1)
            if 'd' in _pressed_keys:
                if y_right and y_right <= br:
                    vy += SPEED
                else:
                    print("[KB] right bound reached")
                    time.sleep(0.1)
            if 'a' in _pressed_keys:
                if y_right and y_right >= bl:
                    vy -= SPEED
                else:
                    print("[KB] left bound reached")
                    time.sleep(0.1)

            # yaw – use a fixed rate, not accumulating each iteration
            if 'q' in _pressed_keys:
                yaw_rate += -np.deg2rad(YAW_RATE)
            elif 'e' in _pressed_keys:
                yaw_rate += np.deg2rad(YAW_RATE)

            # vertical velocity (altitude hold when vz==0)
            if 'h' in _pressed_keys:
                if -mavc.get_pose()[2] < 1.5:  # safety check to prevent flying too high
                    vz -= ALT_SPEED
            if 'n' in _pressed_keys:
                if -mavc.get_pose()[2] > 0.2:  # safety check to prevent crashing into the ground
                    vz += ALT_SPEED

            # always send a velocity setpoint (hover if everything is zero)
            mavc.send_body_offset_ned_vel(vx, vy, vz, yaw_rate=yaw_rate)

            time.sleep(period)

    except KeyboardInterrupt:
        print("[KB] Keyboard interrupt, landing and exiting.")
        mavc.set_mode('LAND')
    finally:
        listener.stop()
        stop_pose_thread()
        # restore terminal echo if we changed it
        if old_settings is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass


if __name__ == '__main__':
    main()