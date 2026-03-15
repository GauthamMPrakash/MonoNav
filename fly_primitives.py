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


# Trajectory execution state (used by the background command thread)
traj_execution_stop_event = threading.Event()


def trajectory_execution_loop(traj, start_pose, command_hz=30):
    """Background thread: execute motion primitive by sending local NED waypoints."""

    period = float(traj['period'])
    times = traj['times']
    x_vals = traj['xvals']
    y_vals = traj['yvals']

    x0, y0, z0, yaw0 = start_pose

    period_s = 1.0 / max(command_hz, 1)
    start_mono = time.monotonic()
    next_command = start_mono

    while not traj_execution_stop_event.is_set():
        elapsed = time.monotonic() - start_mono
        if elapsed < period:
            # Interpolate desired trajectory point at the current time
            x_body = float(np.interp(elapsed, times, x_vals))
            y_body = float(np.interp(elapsed, times, y_vals))

            # Rotate body-frame (forward=x_body, right=y_body) into LOCAL_NED
            # (North, East) using the drone's heading at the start of the primitive.
            # This is a standard 2-D yaw rotation:
            #   north = forward*cos(yaw) - right*sin(yaw)
            #   east  = forward*sin(yaw) + right*cos(yaw)
            cos_yaw = np.cos(yaw0)
            sin_yaw = np.sin(yaw0)
            ned_north_offset = x_body * cos_yaw - y_body * sin_yaw
            ned_east_offset  = x_body * sin_yaw + y_body * cos_yaw

            # Add offset relative to starting pose in LOCAL_NED
            x_target = x0 + ned_north_offset   # LOCAL_NED X = North
            y_target = y0 + ned_east_offset     # LOCAL_NED Y = East
            z_target = z0                        # hold takeoff altitude (NED Z = Down)

            mavc.send_local_ned_pos(x_target, y_target, z_target)
        else:
            # End of trajectory: hold position at starting pose
            mavc.send_local_ned_pos(x0, y0, z0)
            break

        next_command += period_s
        sleep_time = next_command - time.monotonic()
        if sleep_time > 0:
            traj_execution_stop_event.wait(sleep_time)
        else:
            next_command = time.monotonic()


def execute_primitive(traj, start_pose, duration=None, command_hz=10):
    """Execute a single primitive at a fixed command rate using a background thread.

    Args:
        traj: trajectory dict (npz file) loaded from the trajectory library.
        start_pose: (x0, y0, z0, yaw0) local NED pose at the start of the primitive.
        duration: duration of the primitive (seconds). Defaults to traj['period'].
    """

    global traj_execution_stop_event

    if duration is None:
        duration = float(traj['period'])

    traj_execution_stop_event.clear()

    traj_thread = threading.Thread(
        target=trajectory_execution_loop,
        args=(traj, start_pose, command_hz),
        daemon=True,
    )
    traj_thread.start()

    time.sleep(duration)

    traj_execution_stop_event.set()
    traj_thread.join(timeout=1.0)


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

            # read and validate pose at takeoff
            x0 = y0 = z0 = yaw0 = None
            try:
                x0, y0, z0, yaw0, pitch0, roll0 = mavc.get_pose(timeout_s=1.0)
                if any(v is None for v in (x0, y0, z0, yaw0)):
                    raise RuntimeError("pose stream incomplete (None values)")
                print(f"Takeoff pose: x0={x0:.3f}, y0={y0:.3f}, z0={z0:.3f}, yaw0={yaw0:.3f}")
            except Exception as e:
                print(f"Warning: unable to read valid takeoff pose: {e}")
                print("Landing without executing primitive because start pose is invalid.")
                mavc.set_mode('LAND')
                time.sleep(1)
                continue

            print(f"Flying primitive {idx}/{len(amplitudes) - 1} (amp {amplitudes[idx]:.3f})")
            execute_primitive(traj_list[idx], (x0, y0, z0, yaw0))

            # read and print current x,y (LOCAL_NED)
            try:
                x1, y1, z1, yaw1, pitch1, roll1 = mavc.get_pose(timeout_s=1.0)
                if x1 is None or y1 is None:
                    print("Current position: unavailable")
                else:
                    print(f"Current position (NED): x={x1:.3f}, y={y1:.3f}")

                    # If we have a stored takeoff pose, compute the final pose
                    # in RDF (right, down, front) relative to the takeoff origin.
                    if x0 is not None:
                        dx = x1 - x0
                        dy = y1 - y0
                        dz = z1 - z0
                        try:
                            right, down, front = ned_to_rdf(dx, dy, dz, yaw0)
                            # Also compute yaw change relative to takeoff heading
                            yaw_delta = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
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
        print("Interrupted by user — landing and exiting.")
    finally:
        mavc.set_mode('LAND')


if __name__ == '__main__':
    main()
