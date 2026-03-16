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
        self.intrinsic_matrix = np.asarray(intrinsic_matrix)
        self.depth_intrinsic = o3d.core.Tensor(self.intrinsic_matrix)

        # Initialize the VoxelBlockGrid
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=(1, 1, 3),
            voxel_size=3.0 / 64,                    # this sets the resolution of the voxel grid
            block_resolution=1,
            block_count=50000,
            device=device)

    def integration_step(self, color, depth_numpy, cam_pose):
        # Integration Step (TSDF Fusion)
        depth_numpy = depth_numpy.astype(np.uint16)  # Convert to uint16
        depth = o3d.t.geometry.Image(depth_numpy).to(self.device)
        # Open3D frustum indexing expects camera tensors on CPU even when VBG lives on CUDA.
        depth_intrinsic_cpu = self.depth_intrinsic.to(o3d.core.Device("CPU:0"))
        extrinsic_cpu = o3d.core.Tensor(np.linalg.inv(cam_pose))
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, depth_intrinsic_cpu, extrinsic_cpu, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)
        color = o3d.t.geometry.Image(np.asarray(color)).to(self.device)
        color_intrinsic_cpu = o3d.core.Tensor(self.intrinsic_matrix)
        self.vbg.integrate(frustum_block_coords, depth, color, depth_intrinsic_cpu,
                       color_intrinsic_cpu, extrinsic_cpu, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)


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

    depth = depth_anything.infer_image(frame, size)    # as np ndarray, in meters (float32)
    depth = (1000*depth).astype(np.uint16)             # Convert to mm and uint16 for Open3D integration (depth in mm is more standard for TSDF fusion)
    # the above line works as long as depth is ensured to be under 65.535 meters. 
    # In case KITTI is used and you for some reason want to integrate VBG more than this limit (depth_max in config.yml), beware

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


def ned_to_rdf(goal_north, goal_east, goal_down, heading_offset):
    """
    Convert a point from NED coordinates back to the RDF frame used internally.

    This is essentially the inverse of rdf_goal_to_ned()
    """
    # Reverse the rotation applied in rdf_goal_to_ned by rotating by -heading_offset.
    cos_yaw = m.cos(heading_offset)
    sin_yaw = m.sin(heading_offset)

    # Unrotate NED coordinates back to the unrotated RDF orientation
    goal_front = goal_north * cos_yaw + goal_east * sin_yaw
    goal_right = -goal_north * sin_yaw + goal_east * cos_yaw
    goal_down = goal_down

    return goal_right, goal_down, goal_front


"""
Get the global pose from the vehicle, convert to the Open3D frame
ArduPilot frame: (X, Y, Z) is NORTH EAST DOWN (NED)
Open3D frame: (X, Y, Z) is RIGHT DOWN FRONT (RDF)

This function assumes the drone was pointing North at initialization. But since this obviously may not be true, we compute a heading offset initially to convert RDF goal position to actual NED frame. This allows us to work with RDF (assumed to be aligned with NED) coordinates internally (for trajectory primitives and visualization) while still commanding the drone in the correct NED frame according to its actual initial heading. 

This doesn't matter in exploration mode. But in goal-directed navigation, this allows us to specify the goal in RDF coordinates relative to the drone's initial heading, and have it correctly transformed to NED for ArduPilot.
"""
def get_pose_matrix(pos_x, pos_y, pos_z, vehicle_yaw_rad, vehicle_pitch_rad, vehicle_roll_rad):
    # Precompute trigonometric terms for roll, pitch, and yaw
    sr, cr = m.sin(vehicle_roll_rad), m.cos(vehicle_roll_rad)
    sp, cp = m.sin(vehicle_pitch_rad), m.cos(vehicle_pitch_rad)
    sy, cy = m.sin(vehicle_yaw_rad), m.cos(vehicle_yaw_rad)

    # Construct the rotation matrix directly using NumPy for efficiency
    pose = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, pos_y],  # Row 0
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, pos_z],  # Row 1
        [-sp,     cp * sr,               cp * cr,                 pos_x],  # Row 2
        [0,       0,                     0,                           1]   # Homogeneous row
    ])

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
    points = np.array([pose[0:3,3] for pose in poses])  # meters, preallocated
    lines = np.column_stack([np.arange(len(points)-1), np.arange(1, len(points))])

    pose_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    pose_lineset.paint_uniform_color([1, 0, 0]) #optional: change the color here
    return pose_lineset


"""
Load the trajectory primitives (before navigation).
Read a list of motion primitives (trajectories) from a the "trajlib_dir" (trajectory library) directory.
Returns a list of trajectory objects.
"""
def get_trajlist(trajlib_dir):
    # Get the list of sorted .npz files and load them (important for indexing!)
    traj_list = [np.load(os.path.join(trajlib_dir, f)) 
                 for f in sorted(os.listdir(trajlib_dir)) 
                 if f.endswith('.npz')]
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
    max_traj_score = -np.inf # track best trajectory
    min_goal_score = np.inf # track proximity to goal
    max_traj_idx = None # track the index of the best trajectory

    # iterate over the sorted traj linesets
    for traj_idx, traj_linset in enumerate(traj_linesets):
        traj_lineset_copy = copy.deepcopy(traj_linset)
        traj_lineset_copy.transform(camera_position) # transform the lineset (copy) to the camera position
        pts = np.asarray(traj_lineset_copy.points) # meters # extract the points from the lineset
        tmp = distance.cdist(voxel_coords_numpy, pts, "sqeuclidean") # compute the distance between all voxels and all points in the trajectory
        
        # Guard against empty distance matrix (e.g., no voxels mapped yet)
        if tmp.size == 0:
            # No voxels in the scene; allow this trajectory
            if goal_position is not None:
                # With a goal, prefer trajectories; just use the first safe one
                if max_traj_idx is None:
                    max_traj_idx = traj_idx
            else:
                # No goal, just pick the first trajectory
                if max_traj_idx is None:
                    max_traj_idx = traj_idx
            continue
        
        voxel_idx, pt_idx = np.unravel_index(np.argmin(tmp), tmp.shape) # extract indices of the nearest voxel to and nearest point in the trajectory
        nearest_voxel_dist = np.sqrt(tmp[voxel_idx, pt_idx])
        #mavc.printd(f"traj {traj_idx}: nearest_obstacle={nearest_voxel_dist:.3f}m (threshold={dist_threshold}m)")
        if nearest_voxel_dist > dist_threshold:
            # the trajectory meets the dist_threshold criterion
            if goal_position is not None:
                # the trajectory satisfies the dist_threshold; let's compute the goal score
                tmp_to_goal = distance.cdist(goal_position, pts, "sqeuclidean")
                dst_to_goal = np.sqrt(np.min(tmp_to_goal))
                if dst_to_goal < min_goal_score:
                    # we have a trajectory that gets us closer to the goal
                    #mavc.printd("traj %d gets us closer to the goal: %f"%(traj_idx, dst_to_goal))
                    max_traj_idx = traj_idx
                    min_goal_score = dst_to_goal
            else:
                # no goal position, choose the index that maximizes distance from the obstacles
                if max_traj_score < nearest_voxel_dist:
                    # we have found a trajectory that gets us closer to goal
                    max_traj_idx = traj_idx
                    max_traj_score = nearest_voxel_dist

    # Do not force an immediate stop here. Let the caller (`mononav.py`) handle
    # transient cases where `max_traj_idx` is None (e.g., hover and retry).
    return max_traj_idx


# ── Depth-image BendyRuler planner ─────────────────────────────────────────
#
# Inspired by ArduPilot's AP_OABendyRuler::search_xy_path() but works
# directly on a monocular metric depth image instead of a VoxelBlockGrid.
# No trajectory library is required; the output is a chosen horizontal
# bearing offset that callers convert to velocity / yaw-rate commands.

def _wrap_180(angle_deg):
    """Wrap angle to [-180, 180) degrees."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _depth_sector_clearance(depth_m, col_lo, col_hi, row_lo, row_hi, percentile=10):
    """
    Robust minimum depth (metres) inside a rectangular image sector.

    Uses a low percentile so even a small cluster of near pixels flags an
    obstacle, mirroring BendyRuler's conservative margin check.
    """
    col_lo = int(np.clip(col_lo, 0, depth_m.shape[1] - 1))
    col_hi = int(np.clip(col_hi, 0, depth_m.shape[1] - 1))
    row_lo = int(np.clip(row_lo, 0, depth_m.shape[0] - 1))
    row_hi = int(np.clip(row_hi, 0, depth_m.shape[0] - 1))
    if col_hi <= col_lo or row_hi <= row_lo:
        return 0.0
    sector = depth_m[row_lo:row_hi, col_lo:col_hi]
    if sector.size == 0:
        return 0.0
    return float(np.percentile(sector, percentile))


def _bearing_col_range(bearing_deg, half_width_deg, fx, cx, img_width):
    """
    Map a horizontal bearing offset (degrees, +right) and half-sweep width to
    a pixel column range clipped to [0, img_width-1].
    """
    lo = fx * np.tan(np.radians(bearing_deg - half_width_deg)) + cx
    hi = fx * np.tan(np.radians(bearing_deg + half_width_deg)) + cx
    col_lo = int(np.clip(min(lo, hi), 0, img_width - 1))
    col_hi = int(np.clip(max(lo, hi), 0, img_width - 1))
    return col_lo, col_hi


def _memory_sector_clearance(points_cam, bearing_deg, half_width_deg):
    """Return the minimum Euclidean distance of remembered points in a bearing sector."""
    if points_cam is None or len(points_cam) == 0:
        return np.inf

    pts = np.asarray(points_cam, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.inf

    forward_mask = pts[:, 2] > 0.0
    if not np.any(forward_mask):
        return np.inf
    pts = pts[forward_mask]

    bearings = np.degrees(np.arctan2(pts[:, 0], pts[:, 2]))
    sector_mask = np.abs(bearings - float(bearing_deg)) <= float(half_width_deg)
    if not np.any(sector_mask):
        return np.inf

    return float(np.min(np.linalg.norm(pts[sector_mask], axis=1)))


def project_depth_obstacles_to_camera_points(
    depth_m,
    fx,
    fy,
    cx,
    cy,
    row_frac_lo=0.2,
    row_frac_hi=0.8,
    max_depth_m=4.0,
    stride_px=16,
    max_points=400,
):
    """
    Downsample a metric depth image into obstacle points in the camera RDF frame.

    The result is used as a short-lived obstacle memory for the depth-based
    BendyRuler planner so hazards remain visible for a short time after they
    leave the current image.
    """
    H, W = depth_m.shape[:2]
    row_lo = int(H * row_frac_lo)
    row_hi = int(H * row_frac_hi)

    rows = np.arange(row_lo, row_hi, max(int(stride_px), 1), dtype=np.int32)
    cols = np.arange(0, W, max(int(stride_px), 1), dtype=np.int32)
    if rows.size == 0 or cols.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    vv, uu = np.meshgrid(rows, cols, indexing='ij')
    sampled_depth = depth_m[vv, uu]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0) & (sampled_depth <= float(max_depth_m))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    d = sampled_depth[valid].astype(np.float32)
    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)

    x = ((u - float(cx)) / float(fx)) * d
    y = ((v - float(cy)) / float(fy)) * d
    z = d
    pts = np.column_stack((x, y, z)).astype(np.float32)

    if pts.shape[0] > int(max_points):
        idx = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=np.int32)
        pts = pts[idx]
    return pts


class TimedObstacleDatabase:
    """Short-lived obstacle memory stored in world coordinates."""

    def __init__(self, timeout_s=1.0, max_points=1200):
        self.timeout_s = float(timeout_s)
        self.max_points = int(max_points)
        self._points_world = np.empty((0, 3), dtype=np.float32)
        self._timestamps = np.empty((0,), dtype=np.float64)

    def _prune(self, now_s):
        if self._timestamps.size == 0:
            return
        keep = (now_s - self._timestamps) <= self.timeout_s
        self._points_world = self._points_world[keep]
        self._timestamps = self._timestamps[keep]

    def update_from_camera_points(self, camera_points, camera_position, now_s=None):
        now_s = time.time() if now_s is None else float(now_s)
        self._prune(now_s)
        if camera_points is None or len(camera_points) == 0:
            return

        pts = np.asarray(camera_points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return

        rot = np.asarray(camera_position[:3, :3], dtype=np.float32)
        trans = np.asarray(camera_position[:3, 3], dtype=np.float32)
        points_world = (pts @ rot.T) + trans

        self._points_world = np.vstack((self._points_world, points_world))
        self._timestamps = np.concatenate((self._timestamps, np.full(points_world.shape[0], now_s, dtype=np.float64)))

        if self._points_world.shape[0] > self.max_points:
            keep_from = self._points_world.shape[0] - self.max_points
            self._points_world = self._points_world[keep_from:]
            self._timestamps = self._timestamps[keep_from:]

    def get_points_camera(self, camera_position, now_s=None, max_distance_m=None):
        now_s = time.time() if now_s is None else float(now_s)
        self._prune(now_s)
        if self._points_world.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        cam_inv = np.linalg.inv(camera_position)
        rot = np.asarray(cam_inv[:3, :3], dtype=np.float32)
        trans = np.asarray(cam_inv[:3, 3], dtype=np.float32)
        points_cam = (self._points_world @ rot.T) + trans

        keep = points_cam[:, 2] > 0.0
        if max_distance_m is not None:
            keep &= np.linalg.norm(points_cam, axis=1) <= float(max_distance_m)
        return points_cam[keep].astype(np.float32)


def depth_bendyruler_plan(
    depth_m,
    fx,
    cx,
    goal_bearing_deg=0.0,
    lookahead_m=3.0,
    safety_margin_m=1.5,
    bearing_inc_deg=5.0,
    max_bearing_deg=85.0,
    row_frac_lo=0.2,
    row_frac_hi=0.8,
    prev_bearing=None,
    bendy_ratio=1.5,
    bendy_angle=75.0,
    clearance_percentile=10,
    memory_points_cam=None,
):
    """
    BendyRuler-style local planner operating directly on a metric depth image.

    Mirrors AP_OABendyRuler::search_xy_path(): sweeps horizontal bearing
    offsets alternately left/right, performs a two-step clearance check, and
    applies a bearing-change resistance heuristic to reduce oscillation.

    Parameters
    ----------
    depth_m : np.ndarray (H, W) float32
        Metric depth image in metres from DepthAnythingV2.
    fx : float
        Horizontal focal length in pixels.
    cx : float
        Principal-point x coordinate in pixels.
    goal_bearing_deg : float
        Desired direction in the camera frame (degrees, positive = right,
        0 = straight ahead).  Pass 0 for undirected / exploration mode.
    lookahead_m : float
        Step-1 threshold: a direction passes if sector min-depth >= lookahead_m
        (≈ OA_BENDYRULER_LOOKAHEAD).
    safety_margin_m : float
        Step-2 threshold applied to the three sub-bearings from the lookahead
        point (≈ _margin_max in BendyRuler).
    bearing_inc_deg : float
        Sweep increment in degrees (≈ OA_BENDYRULER_BEARING_INC_XY = 5°).
    max_bearing_deg : float
        Maximum search angle left/right (≈ 170° search range at 5° steps).
    row_frac_lo, row_frac_hi : float
        Fractional row band of the depth image used for clearance checks
        (avoids floor at the bottom and sky at the top).
    prev_bearing : float or None
        Bearing chosen at the previous planning step (degrees) for the
        bearing-change resistance heuristic.
    bendy_ratio : float
        Clearance ratio below which a large bearing change is resisted
        (≈ OA_BENDYRULER_CONT_RATIO = 1.5).
    bendy_angle : float
        Bearing change (degrees) above which resistance is active
        (≈ OA_BENDYRULER_CONT_ANGLE = 75°).
    clearance_percentile : int
        Depth percentile used as the sector clearance metric.

    Returns
    -------
    chosen_bearing_deg : float
        Horizontal bearing offset (degrees) to steer toward. Positive = right.
    new_prev_bearing : float
        Updated hysteresis bearing to be stored and passed back next call.
    active : bool
        True if avoidance is redirecting away from straight-ahead (matches
        BendyRuler's ``active`` flag used for telemetry / logging).
    best_margin : float
        Clearance (metres) in the chosen direction.
    """
    H, W = depth_m.shape[:2]
    row_lo = int(H * row_frac_lo)
    row_hi = int(H * row_frac_hi)
    half_inc = bearing_inc_deg / 2.0

    best_bearing = None           # best fully-validated direction
    best_bearing_clearance = -np.inf
    best_margin = -np.inf         # overall best clearance for fallback
    best_margin_bearing = 0.0

    n_steps = int(max_bearing_deg / bearing_inc_deg)

    for i in range(n_steps + 1):
        for side in (0, 1):       # 0 = left (negative), 1 = right (positive)
            if i == 0 and side == 1:   # don't double-check centre
                continue
            bearing_test = float(i * bearing_inc_deg * (1.0 if side == 1 else -1.0))

            col_lo, col_hi = _bearing_col_range(bearing_test, half_inc, fx, cx, W)
            depth_clearance = _depth_sector_clearance(
                depth_m, col_lo, col_hi, row_lo, row_hi, clearance_percentile
            )
            memory_clearance = _memory_sector_clearance(memory_points_cam, bearing_test, half_inc)
            clearance = min(depth_clearance, memory_clearance)

            if (clearance > best_margin or
                    (np.isclose(clearance, best_margin) and
                     abs(_wrap_180(bearing_test - goal_bearing_deg)) <
                     abs(_wrap_180(best_margin_bearing - goal_bearing_deg)))):
                best_margin = clearance
                best_margin_bearing = bearing_test

            if clearance < lookahead_m:
                continue

            # ── Step 2: check three sub-bearings at the lookahead distance ──
            # Mirrors BendyRuler's second-stage {0°, +45°, -45°} fan check.
            step2_ok = False
            for delta2 in (0.0, 45.0, -45.0):
                b2 = bearing_test + delta2
                c2_lo, c2_hi = _bearing_col_range(b2, half_inc, fx, cx, W)
                depth_c2 = _depth_sector_clearance(
                    depth_m, c2_lo, c2_hi, row_lo, row_hi,
                    min(clearance_percentile + 10, 50),   # slightly more lenient
                )
                memory_c2 = _memory_sector_clearance(memory_points_cam, b2, half_inc)
                c2 = min(depth_c2, memory_c2)
                if c2 >= safety_margin_m:
                    step2_ok = True
                    break

            if not step2_ok:
                continue

            active_candidate = (i != 0)

            # ── Bearing-change resistance (mirrors resist_bearing_change()) ──
            final_bearing = bearing_test
            final_clearance = clearance
            if (active_candidate and prev_bearing is not None and
                    abs(_wrap_180(prev_bearing - bearing_test)) > bendy_angle and
                    bendy_ratio > 0):
                p_lo, p_hi = _bearing_col_range(
                    _wrap_180(prev_bearing), half_inc, fx, cx, W
                )
                prev_depth_clearance = _depth_sector_clearance(
                    depth_m, p_lo, p_hi, row_lo, row_hi, clearance_percentile
                )
                prev_memory_clearance = _memory_sector_clearance(memory_points_cam, _wrap_180(prev_bearing), half_inc)
                prev_clearance = min(prev_depth_clearance, prev_memory_clearance)
                if prev_clearance > 0 and clearance < bendy_ratio * prev_clearance:
                    # New direction doesn't offer sufficient improvement —
                    # resist the change and keep the previous bearing.
                    final_bearing = prev_bearing
                    final_clearance = prev_clearance

            # Prefer the candidate closest to the goal bearing.
            final_goal_error = abs(_wrap_180(final_bearing - goal_bearing_deg))
            best_goal_error = np.inf if best_bearing is None else abs(_wrap_180(best_bearing - goal_bearing_deg))
            if (best_bearing is None or
                    final_goal_error < best_goal_error or
                    (np.isclose(final_goal_error, best_goal_error) and
                     final_clearance > best_bearing_clearance)):
                best_bearing = final_bearing
                best_bearing_clearance = final_clearance

    # ── Determine output ─────────────────────────────────────────────────────
    if best_bearing is not None:
        chosen = best_bearing
        new_prev_bearing = best_bearing
        active = abs(_wrap_180(chosen)) > bearing_inc_deg / 2.0
        margin_out = best_bearing_clearance
    else:
        # No fully clear path — use best-effort direction (highest margin).
        chosen = best_margin_bearing
        new_prev_bearing = best_margin_bearing
        active = True
        margin_out = best_margin

    return chosen, new_prev_bearing, active, margin_out


def depth_bendyruler_commands(
    depth_m,
    fx,
    cx,
    goal_bearing_deg=0.0,
    forward_speed=0.5,
    yaw_rate_gain=0.8,
    lateral_gain=0.0,
    lookahead_m=3.0,
    safety_margin_m=1.5,
    bearing_inc_deg=5.0,
    max_bearing_deg=85.0,
    row_frac_lo=0.2,
    row_frac_hi=0.8,
    prev_bearing=None,
    bendy_ratio=1.5,
    bendy_angle=75.0,
    clearance_percentile=10,
    memory_points_cam=None,
):
    """
    Full BendyRuler depth planner: convert the chosen bearing into direct
    body-frame velocity / yaw-rate commands.

    Compatible with ``mavc.send_body_offset_ned_vel(fwd, lat, yaw_rate=yr)``.

    Parameters
    ----------
    depth_m : np.ndarray (H, W) float32
        Metric depth image in metres.
    fx, cx : float
        Horizontal focal length and principal-point x (pixels).
    goal_bearing_deg : float
        Desired horizontal direction in camera frame (degrees, +right).
    forward_speed : float
        Body-x (forward) velocity in m/s when a safe path exists.
    yaw_rate_gain : float
        Proportional gain: yaw_rate = gain * chosen_bearing_rad (rad/s).
    lateral_gain : float
        Optional gain for body-y lateral velocity to counter sideslip during
        turns.  Set 0 to disable (default).  Same sign convention as mononav.
    lookahead_m, safety_margin_m : float
        Step-1 and step-2 depth clearance thresholds (metres).
    bearing_inc_deg, max_bearing_deg : float
        Sweep granularity and range.
    row_frac_lo, row_frac_hi : float
        Depth-image row band for clearance checks.
    prev_bearing : float or None
        Previous chosen bearing (degrees) for hysteresis.
    bendy_ratio, bendy_angle : float
        Bearing-change resistance parameters.
    clearance_percentile : int
        Depth percentile used as obstacle clearance metric.

    Returns
    -------
    fwd_speed : float
        Body-x (forward) velocity command (m/s).
    lat_vel : float
        Body-y (lateral) velocity command (m/s).
    yaw_rate : float
        Yaw-rate command (rad/s, positive = clockwise from above).
    new_prev_bearing : float
        Updated bearing state to be stored and passed back next call.
    active : bool
        True if avoidance is actively redirecting.
    margin : float
        Clearance in the chosen direction (metres).
    """
    chosen_deg, new_prev, active, margin = depth_bendyruler_plan(
        depth_m, fx, cx,
        goal_bearing_deg=goal_bearing_deg,
        lookahead_m=lookahead_m,
        safety_margin_m=safety_margin_m,
        bearing_inc_deg=bearing_inc_deg,
        max_bearing_deg=max_bearing_deg,
        row_frac_lo=row_frac_lo,
        row_frac_hi=row_frac_hi,
        prev_bearing=prev_bearing,
        bendy_ratio=bendy_ratio,
        bendy_angle=bendy_angle,
        clearance_percentile=clearance_percentile,
        memory_points_cam=memory_points_cam,
    )

    # No safe path found — hover in place.
    if margin <= 0:
        return 0.0, 0.0, 0.0, new_prev, active, margin

    chosen_rad = np.radians(chosen_deg)
    yaw_rate = yaw_rate_gain * chosen_rad     # proportional yaw toward chosen direction
    lat_vel = lateral_gain * yaw_rate         # optional lateral bias (same sign as mononav)

    return float(forward_speed), float(lat_vel), float(yaw_rate), new_prev, active, margin


def goal_bearing_cam_deg(camera_position, goal_edm):
    """
    Compute horizontal bearing to goal in the camera/body frame.

    Parameters
    ----------
    camera_position : np.ndarray (4, 4)
        Pose matrix from ``get_pose_matrix()`` (RDF frame used by Open3D /
        VoxelBlockGrid: X=right, Y=down, Z=front).
    goal_edm : array-like (3,) or None
        Goal position in the VBG internal frame [E, D, N] as stored in
        ``goal_position[0]`` in mononav.py.  Pass ``None`` for undirected mode
        (returns 0.0, i.e., aim straight ahead).

    Returns
    -------
    float
        Bearing in degrees; positive = right, 0 = straight ahead.
    """
    if goal_edm is None:
        return 0.0
    g = np.array([float(goal_edm[0]), float(goal_edm[1]), float(goal_edm[2]), 1.0])
    g_cam = np.linalg.inv(camera_position) @ g
    # In the RDF camera frame: x_cam = right, z_cam = forward.
    return float(np.degrees(np.arctan2(float(g_cam[0]), float(g_cam[2]))))


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
    scaled_intrinsic = np.array(intrinsic_matrix, copy=True)
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
    cropped_intrinsic = np.array(intrinsic_matrix, copy=True)
    x, y, _, _ = [int(v) for v in roi]
    cropped_intrinsic[0, 2] -= x
    cropped_intrinsic[1, 2] -= y
    return cropped_intrinsic

"""
Transform the raw image.

This involves resizing the image, scaling the camera matrix, and undistorting
and cropping the image.  The undistortion step may be bypassed by setting
``apply_undistort`` to ``False`` (for example when a sensor is already
rectified or when you want to work with raw frames).

Args:
    image: Input colour image (numpy array or array-like).
    mtx: Camera matrix from calibration.
    dist: Distortion coefficients.
    optimal_matrix: Optimal camera matrix after undistortion.
    roi: Region of interest for cropping (x, y, w, h).
    apply_undistort: If False, the original image is returned (cropped if
        ``roi`` is not ``None``); no undistortion is performed.
"""
def transform_image(image, mtx, dist, optimal_matrix, roi, apply_undistort: bool = True):
    # optionally skip undistortion entirely
    if not apply_undistort:
        if roi is not None:
            x, y, w, h = roi
            return image[y:y+h, x:x+w]
        return image

    # cv2 can handle both numpy arrays and other array-like objects efficiently
    transformed_image = cv2.undistort(image if isinstance(image, np.ndarray) else np.asarray(image), 
                                      mtx, dist, None, optimal_matrix)
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
        ctr.set_zoom(1)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        mavc.printd(f"Open3D visualization failed: {e}")

def _pose_thread_worker():
    """Background thread that polls get_pose at configured frequency and stores latest values (no buffering)."""
    global _pose_thread_stop, _pose_latest, _pose_thread_hz
    
    sleep_time = 1.0 / _pose_thread_hz
    while not _pose_thread_stop:
        x, y, z, yaw, pitch, roll = mavc.get_pose()
        with _pose_lock:
            _pose_latest = {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'pitch': pitch, 'roll': roll}
        time.sleep(sleep_time)

def start_pose_thread(frequency_hz=15):
    """Start the background pose thread at specified frequency.
    
    Args:
        frequency_hz: Update frequency in Hz (default: 15.0)
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