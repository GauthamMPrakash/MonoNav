"""Simple keyboard velocity controller for an ArduCopter just using
``send_body_offset_ned_vel``.

Keys:
  w/s  -> forward/backward
  a/d  -> left/right
  q/e  -> yaw left/right
  h/n  -> increase/decrease altitude in 5cm steps
  t    -> takeoff (to altitude defined in config.yml)
  l    -> land (sets mode to LAND)

Holding a direction or yaw key will continuously send the velocity command;
when all movement keys are released the controller immediately sends a
zero-velocity packet to hover.  A timestamp is tracked for each pressed key
and stale entries are purged so a missed release event cannot "stick" a key
internally.

Holding a key continuously will keep sending the associated velocity
command at a fixed rate.  Releasing all movement keys immediately
sends a zero-velocity (hover) command.  Inputs are handled with
``pynput`` so there is no polling delay.

Run this script independently when you want to manually fly the drone.
It does _not_ perform any of the MonoNav planning logic; it is purely a
teleoperation helper.
"""

import time
from pynput import keyboard
from pynput.keyboard import Key

import numpy as np
from pymavlink import mavutil
from utils.utils import load_config
import utils.mavlink_control as mavc

# ---------------------------------------------------------------------------
# configuration constants
# ---------------------------------------------------------------------------
CONFIG_FILE = 'config.yml'
# single speed parameter used for both forward/backward and strafe
SPEED = 0.5              # horizontal speed (m/s); adjustable with arrow keys
YAW_RATE = 30.0          # deg/s when pressing 'q' or 'e'; adjustable with left/right arrows
COMMAND_HZ = 20          # how many velocity commands per second
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
        print("[KB] takeoff request")
        mavc.takeoff(config['height'])
        mavc.en_pose_stream()
        return
    if k == 'l':
        print("[KB] land request")
        mavc.set_mode('LAND')
        return

    # add movement keys to the set so the main loop knows which directions
    # are currently held down.
    if k in ('w', 'a', 's', 'd', 'q', 'e', 'h', 'n'):
        # record press and timestamp; height keys handled separately
        _pressed_keys.add(k)
        _key_timestamps[k] = time.monotonic()


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

    config = load_config(CONFIG_FILE)

    # connect and prepare the vehicle
    mavc.connect_drone(config['IP'], baud=config.get('baud', 115200))
    mavc.set_mode('GUIDED')
    mavc.arm()

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    print("[KB] Keyboard teleop started.  't' takeoff, 'l' land, Ctrl-C to exit.")

    try:
        period = 1.0 / COMMAND_HZ
        while True:
            # clear any stale keys that haven't been updated in >1s
            now = time.monotonic()
            for k, t in list(_key_timestamps.items()):
                if now - t > 1.0:
                    _pressed_keys.discard(k)
                    _key_timestamps.pop(k, None)

            vx = 0.0
            vy = 0.0
            yaw_rate = 0.0
            vz = 0.0

            # linear velocity (using unified SPEED)
            if 'w' in _pressed_keys:
                vx += SPEED
            if 's' in _pressed_keys:
                vx -= SPEED
            if 'd' in _pressed_keys:
                vy += SPEED
            if 'a' in _pressed_keys:
                vy -= SPEED

            # yaw
            if 'q' in _pressed_keys:
                yaw_rate += np.deg2rad(YAW_RATE)
            if 'e' in _pressed_keys:
                yaw_rate -= np.deg2rad(YAW_RATE)

            # vertical velocity (altitude hold when vz==0)
            if 'h' in _pressed_keys:
                vz += ALT_SPEED
            if 'n' in _pressed_keys:
                vz -= ALT_SPEED

            # always send a velocity setpoint (hover if everything is zero)
            mavc.send_body_offset_ned_vel(vx, vy, vz, yaw_rate=yaw_rate)

            time.sleep(period)

    except KeyboardInterrupt:
        print("[KB] Keyboard interrupt, landing and exiting.")
        mavc.set_mode('LAND')
    finally:
        listener.stop()


if __name__ == '__main__':
    main()
