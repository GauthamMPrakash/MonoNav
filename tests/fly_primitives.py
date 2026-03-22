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
import threading
import numpy as np
import os, sys

# Ensure the repository root is on sys.path so we can import `utils` from anywhere
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import utils.mavlink_control as mavc
from utils.utils import load_config, get_trajlist, ned_to_rdf

# load config values
config = load_config('config.yml')

# get trajectories
traj_list = get_trajlist(config['trajlib_dir'])
period = float(traj_list[0]['period'])
# amplitudes stored in the npz file
amplitudes = [float(traj['amplitude']) for traj in traj_list]

# (No keyboard listener) Script prompts for indices via stdin.

def main():
    # connect and takeoff
    mavc.connect_drone(config['IP'], baud=config['baud'])
    mavc.set_mode('GUIDED')
    # enable pose streaming so we can read position after the primitive
    mavc.en_pose_stream()

    print("Enter a trajectory index to fly, or 'q' to quit.")
    prompt = f"Index (0 to {len(amplitudes)-1}) or 'q': "
    try:
        while True:
            try:
                raw = input(prompt).strip()
            except EOFError:
                print("Input stream closed; exiting.")
                break

            if raw.lower() == 'q':
                print("Quitting...")
                break

            try:
                idx = int(raw)
            except Exception:
                print("Invalid input; enter an integer index or 'q'.")
                continue

            if idx < 0 or idx >= len(amplitudes):
                print("Index out of range; try again.")
                continue

            # cycle: arm, takeoff, store takeoff pose, fly, query position, land
            mavc.arm()
            mavc.set_speed(float(traj_list[idx]['forward_speed']))
            mavc.takeoff(config['height'])

           
            # read and print current x,y (LOCAL_NED)
            try:
                x1, y1, z1, yaw1, pitch1, roll1 = mavc.get_pose()
                print(f"Current position (NED): x={x1:.3f}, y={y1:.3f}")

                            print(f"Final pose in RDF (relative to takeoff heading): right={right:.3f}, down={down:.3f}, front={front:.3f}")
                            print(f"Heading delta (rad): {yaw_delta:.3f} (deg: {np.degrees(yaw_delta):.1f})")
                        except Exception as e:
                            print(f"Failed to convert NED->RDF: {e}")
            except Exception as e:
                print(f"Failed to read pose: {e}")

            print("Landing...")
            mavc.set_mode('LAND')
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user. Landing and exiting.")
    finally:
        mavc.set_mode('LAND')


if __name__ == '__main__':
    main()
