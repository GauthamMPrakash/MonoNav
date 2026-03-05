"""
  __  __                   _   _             
 |  \/  | ___  _ __   ___ | \ | | __ ___   __
 | |\/| |/ _ \| '_ \ / _ \|  \| |/ _` \ \ / /
 | |  | | (_) | | | | (_) | |\  | (_| |\ V / 
 |_|  |_|\___/|_| |_|\___/|_| \_|\__,_| \_/  
Copyright (c) 2023 Nate Simon
License: MIT
Authors: Nate Simon and Anirudha Majumdar, Princeton University
Project Page: https://natesimon.github.io/mononav

Helper functions for the MonoNav project.
Functionality should be concentrated here and shared between the scripts.

"""
import cv2
import numpy as np
from matplotlib import colormaps
from scipy.spatial import distance
import os
import open3d as o3d
import open3d.core as o3c
import math as m
import copy
import yaml, json
import time

from . import mavlink_control as mavc  # ArduCopter MAVLink wrappers (relative import)
import queue, threading                # For bufferless video capture

# Background pose thread variables for non-blocking pose polling
_pose_lock = threading.Lock()
_pose_thread = None
_pose_thread_stop = False
_pose_thread_hz = 10.0  # default frequency
_pose_latest = {'x': 0, 'y': 0, 'z': 0, 'yaw': 0, 'pitch': 0, 'roll': 0}


"""
VoxelBlockGrid class (adapted from Open3D) for ease of initialization and integration.
You can read more about the VoxelBlockGrid here:
https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/voxel_block_grid.html
"""
class VoxelBlockGrid:
    def __init__(
        self,
        depth_scale=1000.0,
        depth_max=5.0,
        trunc_voxel_multiplier=8.0,
        device=o3d.core.Device("CUDA:0"),
        intrinsic_matrix=None,
    ):
        # Reconstruction Information
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.trunc_voxel_multiplier = trunc_voxel_multiplier
        self.device = device
        if intrinsic_matrix is None:
            self.camera = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )  # Kinect Intrinsics (default)
            intrinsic_matrix = self.camera.intrinsic_matrix
        self.intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
        self.depth_intrinsic = o3d.core.Tensor(self.intrinsic_matrix, o3d.core.Dtype.Float64)

        # Initialize the VoxelBlockGrid
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=(1, 1, 3),
            voxel_size=3.0 / 64, # this sets the resolution of the voxel grid
            block_resolution=1,
            block_count=50000,
            device=device)

    def integration_step(self, color, depth_numpy, cam_pose):
        # Integration Step (TSDF Fusion)
        depth_numpy = depth_numpy.astype(np.uint16)  # Convert to uint16
        depth = o3d.t.geometry.Image(depth_numpy).to(self.device)
        extrinsic = o3d.core.Tensor(np.linalg.inv(cam_pose), o3d.core.Dtype.Float64)
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, self.depth_intrinsic, extrinsic, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)
        color = o3d.t.geometry.Image(np.asarray(color)).to(self.device)
        color_intrinsic = o3d.core.Tensor(self.intrinsic_matrix, o3d.core.Dtype.Float64).to(self.device)
        self.vbg.integrate(frustum_block_coords, depth, color, self.depth_intrinsic,
                       color_intrinsic, extrinsic, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)


"""
Bufferless VideoCapture, courtesy of Ulrich Stern (https://stackoverflow.com/a/54577746)
Otherwise, a lag builds up in the video stream.
"""
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    self.q = queue.Queue(maxsize=1)
    self.last_frame = None
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      self.last_frame = frame
      if self.q.full():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put_nowait(frame)

  def read(self, timeout=1.0):
    try:
      return self.q.get(timeout=timeout)
    except queue.Empty:
      if self.last_frame is not None:
        return self.last_frame
      raise RuntimeError("VideoCapture timeout: no frame available")


"""
Compute depth from an RGB image using DepthAnythingV2
Returns depth_numpy (uint16 in mm), depth_colormap (for visualization)
"""
# cmap = colormaps.get_cmap('Spectral')
def compute_depth(frame, depth_anything, size, make_colormap=True):

    depth = depth_anything.infer_image(frame, size)  # as np ndarray, in meters (float32)
    depth = (1000*depth).astype(np.uint16)             # Convert to mm and uint16 for Open3D integration (depth in mm is more standard for TSDF fusion)

    if make_colormap:
        depth_colormap = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #depth_colormap = (cmap(depth_colormap)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
    else:
        depth_colormap = None

    return depth, depth_colormap


"""
Convert a goal position from RDF (Right-Down-Front) frame to NED (North-East-Down) frame.

The RDF frame is anchored to the drone's heading at startup (after heading_offset_init() is called).
The NED frame is the absolute navigation frame from ArduPilot.

Args:
    goal_right: goal position in RDF right direction (meters)
    goal_down: goal position in RDF down direction (meters)
    goal_front: goal position in RDF front direction (meters)

Returns:
    A tuple (goal_north, goal_east, goal_down) in NED frame
"""
def rdf_goal_to_ned(goal_right, goal_down, goal_front, heading_offset):
    
    # Coordinate frame transformation (without rotation):
    # RDF (right, down, front) -> NED (north, east, down)
    # right (X_RDF) -> east (Y_NED)
    # down (Y_RDF) -> down (Z_NED)  
    # front (Z_RDF) -> north (X_NED)
    
    goal_ned_x_unrotated = goal_front
    goal_ned_y_unrotated = goal_right
    goal_ned_z = goal_down
    
    # Apply yaw rotation to account for heading offset.
    # The heading_offset is the absolute yaw at takeoff.
    # RDF coordinates are relative to the drone's initial heading,
    # so we need to rotate them by heading_offset to align with NED.
    cos_yaw = m.cos(heading_offset)
    sin_yaw = m.sin(heading_offset)
    
    goal_ned_x = goal_ned_x_unrotated * cos_yaw - goal_ned_y_unrotated * sin_yaw
    goal_ned_y = goal_ned_x_unrotated * sin_yaw + goal_ned_y_unrotated * cos_yaw
    
    return goal_ned_x, goal_ned_y, goal_ned_z

"""
Get the global pose from the vehicle, convert to the Open3D frame
ArduPilot frame: (X, Y, Z) is NORTH EAST DOWN (NED)
Open3D frame: (X, Y, Z) is RIGHT DOWN FRONT (RDF)

This function assumes the drone was pointing North at initialization. But since this obviously may not be true, we compute a heading offset initially to convert RDF goal position to actual NED frame. This allows us to work with RDF (assumed to be aligned with NED) coordinates internally (for trajectory primitives and visualization) while still commanding the drone in the correct NED frame according to its actual initial heading. 

This doesn't matter in exploration mode. But in goal-directed navigation, this allows us to specify the goal in RDF coordinates relative to the drone's initial heading, and have it correctly transformed to NED for ArduPilot.
"""
def get_pose_matrix(pos_x, pos_y, pos_z, vehicle_yaw_rad, vehicle_pitch_rad, vehicle_roll_rad):
    # Trig terms for roll/pitch/yaw (radians)
    sr, cr = m.sin(vehicle_roll_rad), m.cos(vehicle_roll_rad)
    sp, cp = m.sin(vehicle_pitch_rad), m.cos(vehicle_pitch_rad)
    sy, cy = m.sin(vehicle_yaw_rad), m.cos(vehicle_yaw_rad)

    # Rotation in NED for extrinsic xyz(roll, pitch, yaw)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    # Build homogeneous transform directly in RDF frame.
    # NED->RDF corresponds to index permutation [1, 2, 0] for rows and cols.
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = r11
    pose[0, 1] = r12
    pose[0, 2] = r10
    pose[1, 0] = r21
    pose[1, 1] = r22
    pose[1, 2] = r20
    pose[2, 0] = r01
    pose[2, 1] = r02
    pose[2, 2] = r00

    # Position: AP NED (north, east, down) -> RDF (right, down, front)
    pose[0, 3] = pos_y
    pose[1, 3] = pos_z
    pose[2, 3] = pos_x
    return pose
    
"""
Load the poses (after navigation, for analysis) from the posedir.
Returns a list of pose arrays.
"""
def poses_from_posedir(posedir):
    poses = []
    pose_files = [name for name in os.listdir(posedir) if os.path.isfile(os.path.join(posedir, name)) and name.endswith(".txt")]
    pose_files = sorted(pose_files)

    for pose_file in pose_files:
        cam_pose = np.loadtxt(posedir +"/"+pose_file)
        poses.append(cam_pose)
    return poses

"""
Convert a list of poses (after navigation, for analysis) into a trajectory lineset.
This object is used to visualize the trajectory in Open3D.
Returns a list of of lineset objects representing the camera's pose.
"""
def get_poses_lineset(poses):
    points = []
    lines = []
    for pose in poses:
        position = pose[0:3,3] # meters
        points.append(position)
        lines.append([len(points)-1, len(points)])

    pose_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines[:-1]),
    )
    pose_lineset.paint_uniform_color([1, 0, 0]) #optional: change the color here
    return pose_lineset

"""
Load the trajectory primitives (before navigation).
Read a list of motion primitives (trajectories) from a the "trajlib_dir" (trajectory library) directory.
Returns a list of trajectory objects.
"""
def get_trajlist(trajlib_dir):
    # Get the list of files in the directory
    file_list = os.listdir(trajlib_dir)
    # Filter only .npz files
    npz_files = [file for file in file_list if file.endswith('.npz')]
    # Sort the list of .npz files - important for indexing!
    sorted_files = sorted(npz_files)
    # Iterate over the sorted list of .npz files
    traj_list = []
    for trajfile in sorted_files:
        file_path = os.path.join(trajlib_dir, trajfile)
        traj_list.append(np.load(file_path))
    
    return traj_list

"""
Convert the trajectory list into a list of trajectory linesets.
These are used for visualizing the possible trajectories at each step.
Returns a list of trajectory lineset objects.
"""
def get_traj_linesets(traj_list):
    traj_linesets = []
    amplitudes = []
    for traj in traj_list:
        # traj_dict = {key: traj[key] for key in traj.files}
        z_tsdf = traj['x_sample']
        x_tsdf = traj['y_sample']
        points = []
        lines = []
        for i in range(len(x_tsdf)):
            points.append([x_tsdf[i], 0, z_tsdf[i]])
            lines.append([len(points)-1, len(points)])
        traj_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines[:-1]),
        )
        traj_linesets.append(traj_lineset)
        amplitudes.append(traj['amplitude'])
        # get traj info
        period = traj['period']
        forward_speed = traj['forward_speed']

    return traj_linesets, period, forward_speed, amplitudes


"""
MonoNav Planner: Return the chosen trajectory index given the current position, current reconstruction, trajectory library, and goal position.
"""
def choose_primitive(vbg, camera_position, traj_linesets, goal_position, dist_threshold, filterYvals, filterWeights, filterTSDF, weight_threshold):

    # Boolean for stopping criteria
    shouldStop = False

    # Get weights and tsdf values from the voxel block grid
    weights = vbg.attribute("weight").reshape((-1))
    tsdf = vbg.attribute("tsdf").reshape((-1))
    # Get the voxel_coords, voxel_indices
    voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices()

    # IMPORTANT
    # Use voxel_indices to rearrange weights and tsdf to match voxel_coords
    # Otherwise, the ordering of voxels from the hashmap is non-deterministic
    weights = weights[voxel_indices]
    tsdf = tsdf[voxel_indices]

    # Generate mask to filter out y values (vertical) (+y is DOWN)
    # This is useful to filter out the floor, and avoid obstacles in-plane
    if filterYvals:
        mask = voxel_coords[:, 1] < -0.3
        # Apply mask to voxel_coords and weights
        voxel_coords = voxel_coords[mask]
        weights = weights[mask]
        tsdf = tsdf[mask]

    # Generate mask to filter by weights
    # This rejects voxels below a certain weight threshold
    if filterWeights:
        mask = weights > weight_threshold
        # Apply mask to voxel_coords and weights
        voxel_coords = voxel_coords[mask,:]
        tsdf = tsdf[mask]

    # Generate mask to filter by tsdf value
    if filterTSDF:
        # Generate mask to filter by tsdf values
        mask = tsdf < 0.0
        voxel_coords = voxel_coords[mask,:]

    # transfer to cpu for cdist
    voxel_coords_numpy = voxel_coords.cpu().numpy()

    # NOW WE HAVE A FILTERED SET OF VOXELS THAT REPRESENT OBSTACLES
    # NEXT, WE DETERMINE THE BEST TRAJECTORY ACCORDING TO A COST FUNCTION

    # Initialize scoring variables to evaluate the trajectories
    max_traj_idx = None # track the index of the best trajectory
    straight_idx = len(traj_linesets) // 2  # index of the straight trajectory

    # If no obstacle voxels remain after filtering, avoid argmin on empty arrays.
    # In this case, prefer goal-directed trajectory (or straight trajectory if no goal).
    if voxel_coords_numpy.size == 0:
        if len(traj_linesets) == 0:
            return True, None

        if goal_position is not None:
            goal_scores = []
            for traj_idx, traj_linset in enumerate(traj_linesets):
                traj_lineset_copy = copy.deepcopy(traj_linset)
                traj_lineset_copy.transform(camera_position)
                pts = np.asarray(traj_lineset_copy.points)
                if pts.size == 0:
                    continue
                tmp_to_goal = distance.cdist(goal_position, pts, "sqeuclidean")
                dst_to_goal = np.sqrt(np.min(tmp_to_goal))
                goal_scores.append((traj_idx, dst_to_goal))

            if len(goal_scores) == 0:
                return False, straight_idx

            min_goal_dist = min(score[1] for score in goal_scores)
            goal_tolerance = 0.5
            candidates = [score for score in goal_scores if score[1] <= min_goal_dist + goal_tolerance]
            max_traj_idx = min(candidates, key=lambda score: abs(score[0] - straight_idx))[0]
            return False, max_traj_idx

        return False, straight_idx

    # First pass: collect all safe trajectories (those that clear obstacles)
    safe_trajectories = []  # list of (traj_idx, nearest_voxel_dist, goal_dist)
    
    for traj_idx, traj_linset in enumerate(traj_linesets):
        traj_lineset_copy = copy.deepcopy(traj_linset)
        traj_lineset_copy.transform(camera_position) # transform the lineset (copy) to the camera position
        pts = np.asarray(traj_lineset_copy.points) # meters # extract the points from the lineset
        if pts.size == 0:
            continue
        tmp = distance.cdist(voxel_coords_numpy, pts, "sqeuclidean") # compute the distance between all voxels and all points in the trajectory
        if tmp.size == 0:
            continue
        voxel_idx, pt_idx = np.unravel_index(np.argmin(tmp), tmp.shape) # extract indices of the nearest voxel to and nearest point in the trajectory
        nearest_voxel_dist = np.sqrt(tmp[voxel_idx, pt_idx])
        
        if nearest_voxel_dist > dist_threshold:
            # This trajectory is safe (clears obstacles)
            if goal_position is not None:
                tmp_to_goal = distance.cdist(goal_position, pts, "sqeuclidean")
                dst_to_goal = np.sqrt(np.min(tmp_to_goal))
            else:
                dst_to_goal = None
            safe_trajectories.append((traj_idx, nearest_voxel_dist, dst_to_goal))
    
    # Second pass: select best trajectory among safe ones
    if len(safe_trajectories) > 0:
        if goal_position is not None:
            # Find the minimum goal distance among safe trajectories
            min_goal_dist = min(t[2] for t in safe_trajectories)
            # Allow trajectories within 0.5m of the best goal distance (tolerance for preferring straighter paths)
            goal_tolerance = 0.5
            candidates = [t for t in safe_trajectories if t[2] <= min_goal_dist + goal_tolerance]
            # Among candidates, prefer the one closest to straight (index closest to middle)
            max_traj_idx = min(candidates, key=lambda t: abs(t[0] - straight_idx))[0]
        else:
            # No goal position: among safe trajectories, prefer the one closest to straight
            max_traj_idx = min(safe_trajectories, key=lambda t: abs(t[0] - straight_idx))[0]

    if max_traj_idx is None:
        # No trajectory meets the dist_threshold criterion, crazyflie should stop.
        shouldStop = True
    return shouldStop, max_traj_idx

"""
Load config.yml file
"""
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

"""
Read in the intrinsics.json file and return the camera matrix and distortion coefficients
"""
def get_calibration_values(camera_calibration_path):
    # Load the camera calibration file
    with open(camera_calibration_path, "r") as json_file:
        data = json.load(json_file)
    mtx = np.array(data['camera_matrix'])
    dist = np.array(data['dist_coeffs'])
    opt_mtx = np.array(data['refined_matrix'])
    roi = np.array(data['roi'])
    return mtx, dist, opt_mtx, roi


def get_calibration_resolution(camera_calibration_path):
    with open(camera_calibration_path, "r") as json_file:
        data = json.load(json_file)
    resolution = data.get("resolution", None)
    if resolution is None or len(resolution) != 2:
        return None, None
    return int(resolution[0]), int(resolution[1])


def scale_intrinsics(intrinsic_matrix, scale_x, scale_y):
    scaled_intrinsic = np.array(intrinsic_matrix, dtype=np.float64, copy=True)
    scaled_intrinsic[0, 0] *= scale_x
    scaled_intrinsic[1, 1] *= scale_y
    scaled_intrinsic[0, 2] *= scale_x
    scaled_intrinsic[1, 2] *= scale_y
    return scaled_intrinsic


def adjust_intrinsics_to_frame_size(mtx, dist, optimal_mtx, roi, frame_width, frame_height, calib_width, calib_height):
    """
    Adjust camera intrinsics and ROI to match the actual frame dimensions.
    If frame dimensions differ from calibration dimensions, scales the intrinsics accordingly.
    
    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
        optimal_mtx: Optimal camera matrix (after undistortion)
        roi: Region of interest [x, y, w, h]
        frame_width: Actual frame width
        frame_height: Actual frame height
        calib_width: Calibration frame width
        calib_height: Calibration frame height
    
    Returns:
        Tuple of (mtx, dist, optimal_mtx, roi) - either scaled or original
    """
    if calib_width is None or calib_height is None:
        return mtx, dist, optimal_mtx, roi
    
    scale_x = frame_width / calib_width
    scale_y = frame_height / calib_height
    
    # Only scale if dimensions differ significantly
    if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
        mtx = scale_intrinsics(mtx, scale_x, scale_y)
        optimal_mtx = scale_intrinsics(optimal_mtx, scale_x, scale_y)
        dist = dist.copy()  # Distortion coefficients don't scale with resolution
        roi = np.array(
            [
                int(round(roi[0] * scale_x)),
                int(round(roi[1] * scale_y)),
                int(round(roi[2] * scale_x)),
                int(round(roi[3] * scale_y)),
            ],
            dtype=np.int32,
        )
    
    return mtx, dist, optimal_mtx, roi


def get_cropped_intrinsics(intrinsic_matrix, roi):
    cropped_intrinsic = np.array(intrinsic_matrix, dtype=np.float64, copy=True)
    x, y, _, _ = [int(v) for v in roi]
    cropped_intrinsic[0, 2] -= x
    cropped_intrinsic[1, 2] -= y
    return cropped_intrinsic

"""
Transform the raw image 
This involves resizing the image, scaling the camera matrix, and undistorting the image.
"""
def transform_image(image, mtx, dist, optimal_matrix, roi):
    transformed_image = cv2.undistort(np.asarray(image), mtx, dist, None, optimal_matrix)
    # Crop the image to the ROI
    x, y, w, h = roi
    transformed_image = transformed_image[y:y+h, x:x+w]

    return transformed_image

"""
Helper function to extract the image frame number from the filename string.
"""
def split_filename(filename):
    return int(filename.split("-")[-1].split(".")[0])

def visualize_pointcloud(pcd_legacy, window_name="Reconstruction"):
    """Visualize a legacy Open3D point cloud, matching mononav.py's Visualizer approach."""
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        vis.add_geometry(pcd_legacy)
        ctr = vis.get_view_control()
        bounds = pcd_legacy.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])       # Z up
        ctr.set_front([0, -1, 0])   # Look along -Y (forward)
        ctr.set_zoom(0.7)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        mavc.printd(f"Open3D visualization failed: {e}")

def _pose_thread_worker():
    """Background thread that polls get_pose at configured frequency and stores latest values (no buffering)."""
    global _pose_thread_stop, _pose_latest, _pose_thread_hz
    
    sleep_time = 1.0 / _pose_thread_hz
    while not _pose_thread_stop:
        try:
            x, y, z, yaw, pitch, roll = mavc.get_pose()
            with _pose_lock:
                _pose_latest = {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'pitch': pitch, 'roll': roll}
            time.sleep(sleep_time)
        except Exception as e:
            mavc.printd(f"Error in pose thread: {e}")
            time.sleep(sleep_time)

def start_pose_thread(frequency_hz=10.0):
    """Start the background pose thread at specified frequency.
    
    Args:
        frequency_hz: Update frequency in Hz (default: 10.0)
    """
    global _pose_thread, _pose_thread_stop, _pose_thread_hz
    
    if _pose_thread and _pose_thread.is_alive():
        mavc.printd("Pose thread already running")
        return
    
    _pose_thread_hz = frequency_hz
    _pose_thread_stop = False
    _pose_thread = threading.Thread(target=_pose_thread_worker, daemon=True)
    _pose_thread.start()
    mavc.printd(f"Pose thread started ({frequency_hz}Hz, non-blocking)")

def get_latest_pose():
    """Get the latest pose without blocking. Returns (x, y, z, yaw, pitch, roll)."""
    with _pose_lock:
        return (_pose_latest['x'], _pose_latest['y'], _pose_latest['z'],
                _pose_latest['yaw'], _pose_latest['pitch'], _pose_latest['roll'])

def stop_pose_thread():
    """Stop the background pose thread."""
    global _pose_thread_stop, _pose_thread
    _pose_thread_stop = True
    if _pose_thread and _pose_thread.is_alive():
        _pose_thread.join(timeout=1.0)
    mavc.printd("Pose thread stopped")