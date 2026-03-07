"""
Simple utility to sequentially execute every motion primitive in the
library.  After the script connects and takes off, it sits idle until you
press the **n** key.  Each press causes the next primitive in the ordered
list to be executed once, then the copter pauses again.  You can hit **q**
at any time to land and exit.

This is useful for manually inspecting the behaviour of individual
primitives without running the full planner.

Usage:
    python fly_primitives.py

The script uses the same configuration values as `mononav.py` (forward
speed, gains, trajlib directory, etc.).  It does _not_ perform any
depth/reconstruction or planning logic.
"""

import time
import numpy as np
from pynput import keyboard

import utils.mavlink_control as mavc
from utils.utils import load_config, get_trajlist

# load config values
config = load_config('config.yml')
forward_speed = config['forward_speed']
yvel_gain = config['yvel_gain']
yawrate_gain = config['yawrate_gain']

# get trajectories
traj_list = get_trajlist(config['trajlib_dir'])
period = float(traj_list[0]['period'])
# amplitudes stored in the npz file
amplitudes = [float(traj['amplitude']) for traj in traj_list]

# keyboard state
next_primitive = False
shutdown = False


def on_press(key):
    global next_primitive, shutdown
    try:
        if key.char == 'n':
            next_primitive = True
        elif key.char == 'q':
            shutdown = True
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()


def execute_primitive(amplitude):
    """Run one primitive for exactly `period` seconds."""
    start = time.time()
    while time.time() - start < period:
        t = time.time() - start
        yawrate = amplitude * np.sin(np.pi/period * t)
        yvel = yawrate * yvel_gain
        command_yaw = yawrate * yawrate_gain
        if forward_speed is not None:
            vx = forward_speed
        else:
            vx = traj_list[0]['forward_speed']
        mavc.send_body_offset_ned_vel(vx, yvel, yaw_rate=command_yaw)
        time.sleep(0.1)  # 10 Hz command rate
    # stop motion after primitive
    mavc.send_body_offset_ned_vel(0, 0, yaw_rate=0)


def main():
    global next_primitive, shutdown

    # connect and takeoff
    mavc.connect_drone(config['IP'], baud=config['baud'])
    mavc.set_mode('GUIDED')

    print("Press 'n' to initiate next takeoff/primitive cycle, 'q' to quit.")
    while not shutdown and idx < len(amplitudes):
        # wait for user to request next flight
        while not next_primitive and not shutdown:
            time.sleep(0.1)
        if shutdown:
            break
        next_primitive = False

        # cycle: arm, takeoff, fly, land
        mavc.arm()
        mavc.takeoff(config['height'])
        time.sleep(1)
        idx = int(input(f"Enter trajectory index (0 to {len(amplitudes)-1}): "))
        print(f"Flying primitive {idx}/{len(amplitudes) - 1} (amp {amplitudes[idx]:.3f})")
        execute_primitive(amplitudes[idx])
        print("Landing...")
        mavc.set_mode('LAND')
        time.sleep(1)

    listener.stop()


if __name__ == '__main__':
    main()
