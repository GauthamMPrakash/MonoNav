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
from scipy.spatial import distance, cKDTree
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
        extrinsic_cpu = o3d.core.Tensor(np.linalg.inv(cam_pose), o3d.core.Dtype.Float64)
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, depth_intrinsic_cpu, extrinsic_cpu, self.depth_scale, self.depth_max, self.trunc_voxel_multiplier)
        color = o3d.t.geometry.Image(np.asarray(color)).to(self.device)
        color_intrinsic_cpu = o3d.core.Tensor(self.intrinsic_matrix, o3d.core.Dtype.Float64)
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

    depth = depth_anything.infer_image(frame, size)  # as np ndarray, in meters (float32)
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
Now with support for D* Lite path planning and treating unexplored space as unsafe.

Args:
    vbg: VoxelBlockGrid with reconstruction
    camera_position: Current camera pose (4x4 matrix)
    traj_linesets: List of trajectory linesets to evaluate
    goal_position: Optional goal position
    dist_threshold: Minimum distance to obstacles
    filterYvals: Filter vertical values
    filterWeights: Filter by weight threshold
    filterTSDF: Filter by TSDF values
    weight_threshold: TSDF weight threshold
    exploration_grid: Optional ExplorationGrid to treat unexplored space as obstacles
    unexplored_penalty_dist: Distance penalty for unexplored regions (meters)
    dstar_planner: Optional DStarLitePlanner for global path planning
    use_dstar: Whether to use D* Lite for goal-directed navigation
"""
def choose_primitive(vbg, camera_position, traj_linesets, goal_position, dist_threshold, 
                    filterYvals, filterWeights, filterTSDF, weight_threshold, 
                    exploration_grid=None, unexplored_penalty_dist=0.5,
                    dstar_planner=None, use_dstar=True,
                    planner_local_radius_m=None):

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

    # transfer to cpu for distance queries
    voxel_coords_numpy = voxel_coords.cpu().numpy()

    # Optional local planning window around current camera position.
    if planner_local_radius_m is not None and planner_local_radius_m > 0 and voxel_coords_numpy.shape[0] > 0:
        cam_xyz = camera_position[0:3, 3].reshape(1, 3)
        d2 = np.sum((voxel_coords_numpy - cam_xyz) ** 2, axis=1)
        voxel_coords_numpy = voxel_coords_numpy[d2 <= planner_local_radius_m ** 2]

    # Build once; prevents allocating huge cdist matrices per trajectory.
    obstacle_tree = cKDTree(voxel_coords_numpy) if voxel_coords_numpy.shape[0] > 0 else None

    # D* Lite planning (if enabled and planner provided)
    dstar_path = None
    if use_dstar and dstar_planner is not None and goal_position is not None:
        camera_pos = np.asarray(camera_position[0:3, 3], dtype=np.float64).reshape(3)
        goal_pos = np.asarray(goal_position, dtype=np.float64).reshape(-1)[:3]
        try:
            dstar_path = dstar_planner.plan_to_goal(
                camera_pos, goal_pos, exploration_grid, 
                voxel_coords_numpy, camera_position, weight_threshold
            )
        except Exception as e:
            mavc.printd(f"[WARNING] D* Lite planning failed: {e}, falling back to greedy")

    # NOW WE HAVE A FILTERED SET OF VOXELS THAT REPRESENT OBSTACLES
    # NEXT, WE DETERMINE THE BEST TRAJECTORY ACCORDING TO A COST FUNCTION

    # Initialize scoring variables to evaluate the trajectories
    max_traj_score = -np.inf # track best trajectory
    min_goal_score = np.inf # track proximity to goal
    max_traj_idx = None # track the index of the best trajectory

    rot = camera_position[0:3, 0:3]
    trans = camera_position[0:3, 3]

    # iterate over trajectory library
    for traj_idx, traj_linset in enumerate(traj_linesets):
        # Faster than deepcopy+transform for every primitive.
        base_pts = np.asarray(traj_linset.points)
        pts = base_pts @ rot.T + trans
        
        # Check collision with KNOWN obstacles
        nearest_obstacle_dist = np.inf
        if obstacle_tree is not None and pts.shape[0] > 0:
            dists, _ = obstacle_tree.query(pts, k=1)
            nearest_obstacle_dist = float(np.min(dists))
        
        # Check collision with UNEXPLORED regions
        nearest_unexplored_dist = np.inf
        if exploration_grid is not None:
            grid_idx = ((pts - exploration_grid.grid_origin) / exploration_grid.cell_size).astype(np.int32)
            in_bounds = np.all((grid_idx >= 0) & (grid_idx < exploration_grid.grid_dim), axis=1)

            # Out-of-bounds trajectory points are treated as unknown/unsafe.
            if not np.all(in_bounds):
                nearest_unexplored_dist = 0
            else:
                explored_vals = exploration_grid.exploration_grid[
                    grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]
                ]
                if np.any(explored_vals == 0):
                    nearest_unexplored_dist = 0
        
        # Overall nearest obstacle distance (considering both known and unexplored)
        if nearest_unexplored_dist < unexplored_penalty_dist:
            # Trajectory passes too close to unexplored regions; reject it
            continue
        
        # Take minimum distance (more conservative)
        nearest_total_dist = min(nearest_obstacle_dist, nearest_unexplored_dist if nearest_unexplored_dist < np.inf else nearest_obstacle_dist)
        
        #mavc.printd(f"traj {traj_idx}: nearest_obstacle={nearest_obstacle_dist:.3f}m, nearest_unexplored={nearest_unexplored_dist:.3f}m")
        if nearest_total_dist > dist_threshold:
            # the trajectory meets the dist_threshold criterion
            
            # Score trajectory based on progress along D* Lite path (if available)
            if dstar_path is not None and len(dstar_path) > 0:
                # Find closest point on D* path to trajectory endpoint
                traj_endpoint = pts[-1]
                path_distances = [np.linalg.norm(np.array(p) - traj_endpoint) for p in dstar_path]
                progress_score = -min(path_distances)  # Negative distance = reward
                
                # Larger (closer to zero) is better, e.g. -0.2 beats -1.0.
                if progress_score > max_traj_score:
                    max_traj_idx = traj_idx
                    max_traj_score = progress_score
            
            elif goal_position is not None:
                # Fallback: greedy goal-directed selection (no D* path available)
                tmp_to_goal = distance.cdist(goal_position, pts, "sqeuclidean")
                dst_to_goal = np.sqrt(np.min(tmp_to_goal))
                if dst_to_goal < min_goal_score:
                    # we have a trajectory that gets us closer to the goal
                    #mavc.printd("traj %d gets us closer to the goal: %f"%(traj_idx, dst_to_goal))
                    max_traj_idx = traj_idx
                    min_goal_score = dst_to_goal
            else:
                # no goal position, choose the index that maximizes distance from the obstacles (exploration mode)
                if max_traj_score < nearest_total_dist:
                    # we have found a trajectory that gets us closer to goal
                    max_traj_idx = traj_idx
                    max_traj_score = nearest_total_dist

    # Do not force an immediate stop here. Let the caller (`mononav.py`) handle
    # transient cases where `max_traj_idx` is None (e.g., hover and retry).
    return max_traj_idx


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


"""
ExplorationGrid: Track which 3D regions have been explored (observed by the camera).
Treats unexplored voxels as unsafe obstacles to prevent collisions with occluded obstacles.
"""
class ExplorationGrid:
    """
    Tracks explored regions of space based on camera field of view.
    Cells that haven't been observed are marked as unexplored (unsafe).
    
    Args:
        grid_size: Physical size of the grid in meters (e.g., 20m x 20m x 20m)
        cell_size: Size of each grid cell in meters (e.g., 0.1m)
        max_depth: Maximum depth for which we consider space "explored"
    """
    def __init__(self, grid_size=20.0, cell_size=0.1, max_depth=15.0):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.max_depth = max_depth
        # Grid dimensions
        self.grid_dim = int(grid_size / cell_size)
        # Exploration grid: 0=unexplored, 1=explored, -1=explored and empty
        # We use int8 to save memory
        self.exploration_grid = np.zeros((self.grid_dim, self.grid_dim, self.grid_dim), dtype=np.int8)
        self.grid_origin = None  # Will be set at first update
    
    def update_from_depth(self, depth_numpy, camera_position, intrinsics, max_depth_pixels=None):
        """
        Update exploration grid based on depth image and camera pose.
        Marks observed regions as explored.
        
        Args:
            depth_numpy: Depth image (H, W) in meters
            camera_position: 4x4 camera pose matrix in RDF frame
            intrinsics: 3x3 camera intrinsic matrix
            max_depth_pixels: Optional cap on number of depth points/rays used per update.
        """
        if self.grid_origin is None:
            # Initialize origin at camera position
            cam_pos = camera_position[0:3, 3]
            self.grid_origin = cam_pos - self.grid_size / 2.0
        
        camera_center = camera_position[0:3, 3]
        
        # Convert depth to 3D points in camera frame
        h, w = depth_numpy.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Create pixel coordinates
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)
        
        # Convert to camera coordinates (in meters)
        depth_m = depth_numpy.astype(np.float32) / 1000.0  # Convert mm to meters
        x_cam = (uu - cx) * depth_m / fx
        y_cam = (vv - cy) * depth_m / fy
        z_cam = depth_m
        
        # Only consider valid depth values
        valid_mask = (depth_m > 0) & (depth_m < self.max_depth)
        
        # Transform points to world frame (RDF)
        rotation = camera_position[0:3, 0:3]
        translation = camera_position[0:3, 3]
        
        # Stack coordinates for batch transformation
        points_cam = np.stack([x_cam[valid_mask], y_cam[valid_mask], z_cam[valid_mask]], axis=1)
        points_world = points_cam @ rotation.T + translation
        
        # Mark cells as explored
        cell_indices = ((points_world - self.grid_origin) / self.cell_size).astype(np.int32)
        
        # Filter points within grid bounds
        in_bounds = np.all((cell_indices >= 0) & (cell_indices < self.grid_dim), axis=1)
        cell_indices = cell_indices[in_bounds]

        # Optional compute cap: subsample points/rays to keep runtime predictable.
        if max_depth_pixels is not None and max_depth_pixels > 0 and len(cell_indices) > max_depth_pixels:
            stride = max(1, len(cell_indices) // int(max_depth_pixels))
            cell_indices = cell_indices[::stride]
        
        # Mark explored cells
        self.exploration_grid[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]] = 1
        
        # Mark ray-casting: from camera to each sampled point marks space as empty (explored)
        start_idx = ((camera_center - self.grid_origin) / self.cell_size).astype(np.int32)
        for end_idx in cell_indices:
            
            # Bresenham-like line drawing (simple version)
            steps = max(abs(end_idx[0] - start_idx[0]), 
                       abs(end_idx[1] - start_idx[1]), 
                       abs(end_idx[2] - start_idx[2])) + 1
            
            if steps > 1:
                line_indices = np.linspace(start_idx, end_idx, steps, dtype=np.int32)
                line_in_bounds = np.all((line_indices >= 0) & (line_indices < self.grid_dim), axis=1)
                line_indices = line_indices[line_in_bounds]
                self.exploration_grid[line_indices[:, 0], line_indices[:, 1], line_indices[:, 2]] = -1
    
    def get_unexplored_voxels(self, region_bounds=None):
        """
        Get coordinates of unexplored voxels as point cloud.
        Useful for visualizing unexplored regions.
        
        Args:
            region_bounds: Optional bounds [min_x, max_x, min_y, max_y, min_z, max_z] in meters
        
        Returns:
            Array of unexplored voxel coordinates in world frame
        """
        unexplored_mask = self.exploration_grid == 0
        
        if region_bounds is not None:
            # Filter by region
            idx_lower = ((np.array(region_bounds[0:2:2]) - self.grid_origin) / self.cell_size).astype(np.int32)
            idx_upper = ((np.array(region_bounds[1:2:2]) - self.grid_origin) / self.cell_size).astype(np.int32)
            idx_lower = np.maximum(idx_lower, 0)
            idx_upper = np.minimum(idx_upper, self.grid_dim)
            unexplored_mask[0:idx_lower[0], :, :] = False
            unexplored_mask[idx_upper[0]:, :, :] = False
            unexplored_mask[:, 0:idx_lower[1], :] = False
            unexplored_mask[:, idx_upper[1]:, :] = False
            unexplored_mask[:, :, 0:idx_lower[2]] = False
            unexplored_mask[:, :, idx_upper[2]:] = False
        
        indices = np.where(unexplored_mask)
        if len(indices[0]) == 0:
            return np.array([]).reshape(0, 3)
        
        coords = np.stack(indices, axis=1)
        world_coords = coords * self.cell_size + self.grid_origin
        return world_coords


"""
ClosedLoopPrimitiveController: Improved motion primitive tracking with feedback.
Uses position feedback to correct tracking errors during primitive execution.
"""
class ClosedLoopPrimitiveController:
    """
    Executes motion primitives with closed-loop feedback control.
    Tracks the expected trajectory and corrects for deviations.
    
    Args:
        forward_speed: Nominal forward speed (m/s)
        yvel_gain: Gain for lateral velocity (dimensionless)
        yawrate_gain: Gain for yaw rate control (dimensionless)
        kp_lateral: Proportional gain for lateral position error (default: 0.5)
        kp_angular: Proportional gain for angular error (default: 0.3)
    """
    def __init__(self, forward_speed=1.0, yvel_gain=0.5, yawrate_gain=1.0, 
                 kp_lateral=0.5, kp_angular=0.3):
        self.forward_speed = forward_speed
        self.yvel_gain = yvel_gain
        self.yawrate_gain = yawrate_gain
        self.kp_lateral = kp_lateral
        self.kp_angular = kp_angular
        
        # Execution state
        self.start_position = None
        self.start_heading = None
        self.trajectory_start_time = None
        self.period = None
        self.amplitude = None
        self.desired_trajectory = None
        self.trajectory_start_pose = None
    
    def initialize_trajectory(self, start_position, start_heading, period, amplitude, 
                             forward_speed=None):
        """
        Initialize a new trajectory execution.
        
        Args:
            start_position: Initial position [x, y, z]
            start_heading: Initial heading in radians
            period: Trajectory period in seconds
            amplitude: Sinusoidal amplitude for yaw rate
            forward_speed: Optional override for forward speed
        """
        self.start_position = np.array(start_position)
        self.start_heading = start_heading
        self.trajectory_start_time = time.time()
        self.period = period
        self.amplitude = amplitude
        if forward_speed is not None:
            self.forward_speed = forward_speed
        
        # Pre-compute ideal trajectory for reference
        self.desired_trajectory = self._compute_desired_trajectory()
    
    def _compute_desired_trajectory(self, dt=0.01):
        """
        Pre-compute the desired trajectory for closed-loop tracking.
        Returns list of (t, x, y, z, heading) tuples.
        """
        trajectory = []
        num_steps = int(self.period / dt)
        for i in range(num_steps + 1):
            t = i * dt
            if t > self.period:
                t = self.period
            
            # Ideal trajectory: straight forward with sinusoidal yaw
            x = self.forward_speed * t
            y = 0  # Lateral offset computed from yaw rate integration
            z = 0  # Altitude held constant
            heading = self.amplitude * np.sin(np.pi * t / self.period)
            
            trajectory.append((t, x, y, z, heading))
        
        return trajectory
    
    def compute_control(self, current_position, current_heading):
        """
        Compute control commands with closed-loop feedback.
        
        Args:
            current_position: Current position [x, y, z] in RDF frame
            current_heading: Current heading in radians (yaw)
        
        Returns:
            Tuple: (vx, vy, yaw_rate) control commands
        """
        if self.trajectory_start_time is None:
            return self.forward_speed, 0, 0
        
        elapsed = time.time() - self.trajectory_start_time
        
        # Clamp elapsed time to trajectory period
        if elapsed > self.period:
            return 0, 0, 0
        
        # Get desired state from pre-computed trajectory
        desired_heading = self.amplitude * np.sin(np.pi * elapsed / self.period)
        desired_yaw_rate = self.amplitude * (np.pi / self.period) * np.cos(np.pi * elapsed / self.period)
        
        # Compute heading error
        heading_error = desired_heading - current_heading
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Closed-loop yaw rate correction
        yaw_rate = desired_yaw_rate + self.kp_angular * heading_error
        
        # Forward speed (open-loop for now, could add feedback)
        vx = self.forward_speed
        
        # Lateral velocity derived from yaw rate
        yvel = yaw_rate * self.yvel_gain
        
        # Scale yaw rate for ArduCopter
        yaw_rate_command = yaw_rate * self.yawrate_gain
        
        return vx, yvel, yaw_rate_command
    
    def is_complete(self):
        """Check if trajectory execution is complete."""
        if self.trajectory_start_time is None:
            return False
        elapsed = time.time() - self.trajectory_start_time
        return elapsed >= self.period if self.period else False


"""
D* Lite Planner: Incremental shortest path planning for exploration with partial maps.
Optimal for monocular vision where the map is discovered incrementally.

D* Lite efficiently replans when new obstacles are discovered, making it ideal for:
- Unknown/partially-known environments (monocular vision)
- Goal-directed navigation with incremental map updates
- Undirected exploration with frontier-based goals

Reference: Koenig & Likhachev, "D* Lite", IJCAI 2005
"""
class DStarLitePlanner:
    """
    D* Lite path planner for navigation with incremental map updates.
    
    Args:
        grid_size: Physical size of planning grid in meters (e.g., 50m)
        cell_size: Size of each grid cell in meters (e.g., 0.2m)
        cost_obstacle: Cost assigned to obstacles
        cost_free: Cost of traversable space
        k_max_iterations: Max algorithm iterations per replan
    """
    def __init__(self, grid_size=50.0, cell_size=0.2, cost_obstacle=1000.0, 
                 cost_free=1.0, k_max_iterations=1000):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.cost_obstacle = cost_obstacle
        self.cost_free = cost_free
        self.k_max_iterations = k_max_iterations
        
        # Grid setup
        self.grid_dim = int(grid_size / cell_size)
        self.h = self.grid_dim  # heuristic scale
        self.grid_origin = None
        
        # Cost map: 0=free, 1=unexplored (treated as obstacle), 2=obstacle
        self.cost_map = np.ones((self.grid_dim, self.grid_dim, self.grid_dim), dtype=np.float32)
        
        # D* Lite algorithm state
        self.g = {}  # Cost-to-come from goal
        self.rhs = {}  # Lookahead estimate
        self.open_list = []  # Priority queue (key, cell)
        self.start = None
        self.goal = None
        self.last_start = None
        self.km = 0
        self.path = []
        self.replanned = False
        
    def _heuristic(self, a, b):
        """Euclidean heuristic in grid space."""
        diff = np.array(a) - np.array(b)
        return np.linalg.norm(diff) * self.cell_size
    
    def _key(self, s):
        """D* Lite key for cell s."""
        if s == self.goal:
            return (self.g.get(s, np.inf), 0)
        h_val = self._heuristic(s, self.goal) if self.goal is not None else 0
        return (min(self.g.get(s, np.inf), self.rhs.get(s, np.inf)) + h_val + self.km, 
                min(self.g.get(s, np.inf), self.rhs.get(s, np.inf)))

    def _world_to_cell(self, world_pos):
        """Convert world position to hashable integer cell index tuple."""
        if world_pos is None:
            return None
        vec = np.asarray(world_pos, dtype=np.float64).reshape(-1)
        if vec.size < 3:
            return None
        vec = vec[:3]
        idx = ((vec - self.grid_origin) / self.cell_size).astype(np.int32)
        return (int(idx[0]), int(idx[1]), int(idx[2]))
    
    def _get_neighbors(self, cell):
        """Get 6-connected neighbors in 3D grid (considering only horizontal movement)."""
        x, y, z = cell
        neighbors = []
        # 4-connected in horizontal plane (x,z), fixed y (height)
        for dx, dz in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, nz = x + dx, z + dz
            if 0 <= nx < self.grid_dim and 0 <= nz < self.grid_dim:
                neighbors.append((nx, y, nz))
        return neighbors
    
    def _cost(self, from_cell, to_cell):
        """Traversal cost between cells."""
        if self._is_occupied(to_cell):
            return self.cost_obstacle
        return self.cost_free * self._heuristic(from_cell, to_cell)
    
    def _is_occupied(self, cell):
        """Check if cell is occupied or unexplored."""
        x, y, z = cell
        if not (0 <= x < self.grid_dim and 0 <= y < self.grid_dim and 0 <= z < self.grid_dim):
            return True  # Out of bounds is occupied
        return self.cost_map[x, y, z] > 1.5  # > 1 means obstacle or unexplored
    
    def update_map(self, exploration_grid, voxel_coords, camera_position, weight_threshold):
        """
        Update the cost map based on exploration grid and voxel obstacles.
        
        Args:
            exploration_grid: ExplorationGrid instance tracking explored regions
            voxel_coords: Numpy array of known obstacle positions
            camera_position: Current camera pose
            weight_threshold: Obstacle weight threshold
        """
        if self.grid_origin is None:
            # Initialize grid origin
            cam_pos = camera_position[0:3, 3]
            self.grid_origin = cam_pos - self.grid_size / 2.0
        
        # Update grid origin to follow camera (sliding window)
        cam_pos = camera_position[0:3, 3]
        new_origin = cam_pos - self.grid_size / 2.0
        if np.linalg.norm(new_origin - self.grid_origin) > self.cell_size * 2:
            self.grid_origin = new_origin
            self.cost_map = np.ones_like(self.cost_map)  # Reset map
        
        # Mark explored regions as free or unexplored
        if exploration_grid is not None and exploration_grid.grid_origin is not None:
            for i in range(self.grid_dim):
                for j in range(self.grid_dim):
                    for k in range(self.grid_dim):
                        world_pos = np.array([i, j, k]) * self.cell_size + self.grid_origin
                        grid_idx = ((world_pos - exploration_grid.grid_origin) / exploration_grid.cell_size).astype(np.int32)
                        
                        if np.all(grid_idx >= 0) and np.all(grid_idx < exploration_grid.grid_dim):
                            state = exploration_grid.exploration_grid[grid_idx[0], grid_idx[1], grid_idx[2]]
                            if state == -1 or state == 1:  # Explored
                                self.cost_map[i, j, k] = self.cost_free
                            # state == 0 stays as 1 (unexplored/obstacle)
        
        # Mark known obstacles
        if voxel_coords is not None and voxel_coords.shape[0] > 0:
            for obs in voxel_coords:
                cell_idx = ((obs - self.grid_origin) / self.cell_size).astype(np.int32)
                if np.all(cell_idx >= 0) and np.all(cell_idx < self.grid_dim):
                    self.cost_map[cell_idx[0], cell_idx[1], cell_idx[2]] = 2.0
    
    def plan_to_goal(self, start_pos, goal_pos, exploration_grid=None, voxel_coords=None, 
                     camera_position=None, weight_threshold=4.0):
        """
        Plan a path from start to goal using D* Lite.
        
        Args:
            start_pos: Starting position in world coordinates
            goal_pos: Goal position in world coordinates (or None for exploration)
            exploration_grid: Optional ExplorationGrid for map updates
            voxel_coords: Optional known obstacles
            camera_position: Current camera pose
            weight_threshold: Obstacle threshold
            
        Returns:
            Path as list of waypoints, or None if no path exists
        """
        # Convert world coords to grid indices
        if self.grid_origin is None and camera_position is not None:
            self.grid_origin = camera_position[0:3, 3] - self.grid_size / 2.0
        
        if self.grid_origin is None:
            return None
        
        start_cell = self._world_to_cell(start_pos)
        goal_cell = self._world_to_cell(goal_pos)

        if start_cell is None or goal_cell is None:
            return None
        
        # Update map with new obstacle information
        self.update_map(exploration_grid, voxel_coords, camera_position, weight_threshold)
        
        # Initial planning or replan if goal changed
        if self.start != start_cell or self.goal != goal_cell or self.last_start != start_cell:
            self.goal = goal_cell
            self.start = start_cell
            
            # Compute/recompute k_m and costs
            if self.last_start is not None:
                self.km += self._heuristic(self.last_start, self.start)
            
            self.last_start = self.start
            self._initialize_d_star_lite()
            
            # Run D* Lite algorithm
            self._compute_shortest_path()
        
        # Extract path from start to goal
        if self.start in self.g and np.isfinite(self.g[self.start]):
            path = self._extract_path()
            return path
        
        return None
    
    def _initialize_d_star_lite(self):
        """Initialize D* Lite algorithm."""
        self.g = {}
        self.rhs = {self.goal: 0}  # rhs(goal) = 0
        self.open_list = [(self._key(self.goal), self.goal)]
    
    def _compute_shortest_path(self):
        """Main D* Lite search loop."""
        iterations = 0
        while iterations < self.k_max_iterations:
            if not self.open_list:
                break
            
            # Get cell with minimum key
            self.open_list.sort()
            _, u = self.open_list.pop(0)
            
            k_old = self._key(u)
            
            if k_old[0] != float('inf') and self.rhs.get(u, np.inf) > self.g.get(u, np.inf):
                # Over-consistent: g(u) < rhs(u)
                self.g[u] = self.rhs.get(u, np.inf)
                
                # Update neighbors
                for neighbor in self._get_neighbors(u):
                    if neighbor != self.goal:
                        cost = self._cost(neighbor, u)
                        self.rhs[neighbor] = min(self.rhs.get(neighbor, np.inf), 
                                               self.g.get(u, np.inf) + cost)
                    
                    key = self._key(neighbor)
                    if neighbor in [cell for _, cell in self.open_list]:
                        self.open_list.remove((self._key(neighbor), neighbor))
                    
                    if self.g.get(neighbor, np.inf) != self.rhs.get(neighbor, np.inf):
                        self.open_list.append((key, neighbor))
            
            elif self.rhs.get(u, np.inf) < self.g.get(u, np.inf):
                # Under-consistent: rhs(u) < g(u)
                self.g[u] = np.inf
                
                # Update u and neighbors
                for neighbor in self._get_neighbors(u) + [u]:
                    if neighbor != self.goal:
                        cost = self._cost(neighbor, u)
                        self.rhs[neighbor] = min(self.rhs.get(neighbor, np.inf), 
                                               self.g.get(u, np.inf) + cost)
                    
                    key = self._key(neighbor)
                    if neighbor in [cell for _, cell in self.open_list]:
                        self.open_list.remove((self._key(neighbor), neighbor))
                    
                    if self.g.get(neighbor, np.inf) != self.rhs.get(neighbor, np.inf):
                        self.open_list.append((key, neighbor))
            
            iterations += 1
    
    def _extract_path(self):
        """Extract path from start toward goal by greedy descent."""
        path = [self.start]
        current = self.start
        
        for _ in range(self.grid_dim * 3):  # Limit path length
            if current == self.goal:
                break
            
            neighbors = self._get_neighbors(current)
            if not neighbors:
                break
            
            # Greedy descent: choose neighbor with minimum g + cost
            best_neighbor = None
            best_cost = np.inf
            
            for neighbor in neighbors:
                cost = self.g.get(neighbor, np.inf) + self._cost(current, neighbor)
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor
            
            if best_neighbor is None or best_cost == np.inf:
                break
            
            path.append(best_neighbor)
            current = best_neighbor
        
        # Convert grid indices to world coordinates
        world_path = [np.array(cell) * self.cell_size + self.grid_origin for cell in path]
        return world_path
        return elapsed >= self.period