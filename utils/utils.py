r"""
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
import yaml, json
import time

from . import mavlink_control as mavc  # ArduCopter MAVLink wrappers (relative import)
import threading                       # For bufferless video capture and pose threading

_pose_latest = None   # (timestamp, x, y, z, yaw, pitch, roll)
_pose_thread = None
_pose_thread_hz = 15

_stop_event = threading.Event()
_pose_ready = threading.Event()   # signals first pose is available


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
        self.intrinsic_matrix = None
        self.depth_intrinsic = None
        if intrinsic_matrix is not None:
            self.set_intrinsics(intrinsic_matrix)

        # Initialize the VoxelBlockGrid
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=(1, 1, 3),
            voxel_size=3.0 / 64,                    # this sets the resolution of the voxel grid
            block_resolution=1,
            block_count=50000,
            device=device)

    def set_intrinsics(self, intrinsic_matrix):
        self.intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
        self.depth_intrinsic = o3d.core.Tensor(self.intrinsic_matrix, o3d.core.Dtype.Float64)

    def ensure_intrinsics_for_frame(self, frame_width, frame_height):
        if self.intrinsic_matrix is None:
            self.set_intrinsics(get_ideal_intrinsics(frame_width, frame_height))

    def integration_step(self, color, depth_numpy, cam_pose):
        # Integration Step (TSDF Fusion)
        depth_numpy = depth_numpy.astype(np.uint16)  # Convert to uint16
        frame_height, frame_width = depth_numpy.shape[:2]
        self.ensure_intrinsics_for_frame(frame_width, frame_height)
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

Use this for USB receivers when using something like an FPV camera
"""
# import queue
# class VideoCapture:

#   def __init__(self, name):
#     self.cap = cv2.VideoCapture(name)
#     self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     self.q = queue.Queue(maxsize=1)
#     self.last_frame = None
#     t = threading.Thread(target=self._reader)
#     t.daemon = True
#     t.start()

#   # read frames as soon as they are available, keeping only most recent one
#   def _reader(self):
#     while True:
#       ret, frame = self.cap.read()
#       if not ret:
#         break
#       self.last_frame = frame
#       if self.q.full():
#         try:
#           self.q.get_nowait()   # discard previous (unprocessed) frame
#         except queue.Empty:
#           pass
#       self.q.put_nowait(frame)

#   def read(self, timeout=1.0):
#     try:
#       return self.q.get(timeout=timeout)
#     except queue.Empty:
#       if self.last_frame is not None:
#         return self.last_frame
#       raise RuntimeError("VideoCapture timeout: no frame available")
    
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self, timeout=1.0):
        start = time.time()
        while True:
            with self.lock:
                if self.frame is not None:
                    return self.frame
            if time.time() - start > timeout:
                raise RuntimeError("No frame received")
            time.sleep(0.005)

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread is not None:
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
                
"""
Compute depth from an RGB image using DepthAnythingV2
Returns: depth_numpy (uint16 in mm), depth_colormap (for visualization) if make_colormap is True, otherwise None.

Uncomment the cmap line and the currently commented depth_colormap line and comment out the second depth_colormap line for a nicer colormap but adds a slight overhead
"""
# cmap = colormaps.get_cmap('Spectral')
def compute_depth(frame, depth_model, size, make_colormap=True):

    depth = depth_model.infer_image(frame, size)       # as np ndarray, in meters (float32)
    depth = (1000*depth).astype(np.uint16)             # Convert to mm and uint16 for Open3D integration (depth in mm is more standard for TSDF fusion)
    # the above line works as long as depth is ensured to be under 65.535 meters but this shouldn't matter
    # In case KITTI is used and you for some reason want to integrate VBG more than this limit (depth_max in config.yml), beware

    if make_colormap:
        depth_colormap = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
        #depth_colormap = (cmap(depth_colormap)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
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
    
    # Apply yaw rotation to account for heading offset.
    # The heading_offset is the absolute yaw at takeoff.
    # RDF coordinates are relative to the drone's initial heading,
    # so we need to rotate them by heading_offset to align with NED.
    cos_yaw = m.cos(heading_offset)
    sin_yaw = m.sin(heading_offset)
    
    goal_ned_x = goal_front * cos_yaw - goal_right * sin_yaw
    goal_ned_y = goal_front * sin_yaw + goal_right * cos_yaw
    
    return goal_ned_x, goal_ned_y, goal_down


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

    # NED->EDN (considered as RDF for Open3D) basis reorder
    pose = np.array([
        [sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,  sy*cp,  pos_y],
        [cp*sr,             cp*cr,             -sp,    pos_z],
        [cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,  cy*cp,  pos_x],
        [0,                 0,                 0,          1]
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
        x_tsdf = -traj['y_sample']
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
def choose_primitive(vbg, camera_position, traj_linesets, goal_position, dist_threshold, filterYvals, filterWeights, filterTSDF, weight_threshold, DEBUG=False):

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
        voxel_idx, pt_idx = np.unravel_index(np.argmin(tmp), tmp.shape) # extract indices of the nearest voxel to and nearest point in the trajectory
        nearest_voxel_dist = np.sqrt(tmp[voxel_idx, pt_idx])
        if nearest_voxel_dist > dist_threshold:
            # the trajectory meets the dist_threshold criterion
            if goal_position is not None:
                # the trajectory satisfies the dist_threshold; let's compute the goal score
                tmp_to_goal = distance.cdist(goal_position, pts, "sqeuclidean")
                dst_to_goal = np.sqrt(np.min(tmp_to_goal))
                if dst_to_goal < min_goal_score:
                    # we have a trajectory that gets us closer to the goal
                    # print("traj %d gets us closer to the goal: %f"%(traj_idx, dst_to_goal))
                    max_traj_idx = traj_idx
                    min_goal_score = dst_to_goal
            else:
                # no goal position, choose the index that maximizes distance from the obstacles
                if max_traj_score < nearest_voxel_dist:
                    # we have found a trajectory that gets us closer to goal
                    max_traj_idx = traj_idx
                    max_traj_score = nearest_voxel_dist

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
    scaled_intrinsic = np.array(intrinsic_matrix, copy=True)
    scaled_intrinsic[0, 0] *= scale_x
    scaled_intrinsic[1, 1] *= scale_y
    scaled_intrinsic[0, 2] *= scale_x
    scaled_intrinsic[1, 2] *= scale_y
    return scaled_intrinsic


def get_ideal_intrinsics(frame_width, frame_height):
    """
    Return a simple pinhole camera matrix for the given frame size.

    This is only used as a fallback when no calibration intrinsics are available.
    It assumes square pixels, a centered principal point, and a focal length equal
    to the larger image dimension.
    """
    focal_length = float(max(frame_width, frame_height))
    return np.array(
        [
            [focal_length, 0.0, (frame_width - 1) / 2.0],
            [0.0, focal_length, (frame_height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


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
def transform_image(image, mtx=None, dist=None, optimal_matrix=None, roi=None, enable_undistort = True, roi_crop = True):
    transformed_image = image
    if enable_undistort:
        if not roi_crop:
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
        # Open3D renamed `get6_axis_aligned_bounding_box` to `get_axis_aligned_bounding_box`.
        # Use whichever is available for maximum compatibility.
        get_aabb = getattr(pcd_legacy, 'get_axis_aligned_bounding_box', None)
        if get_aabb is not None:
            bounds = get_aabb()
        else:
            bounds = pcd_legacy.get6_axis_aligned_bounding_box()
        center = bounds.get_center()
        ctr.set_lookat(center)
        ctr.set_up([0, 0, 1])       # Z up
        ctr.set_front([0, -1, 0])   # Look along -Y (forward)
        ctr.set_zoom(1)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"[vis] Open3D visualization failed: {e}")

def _pose_thread_worker():
    """Background thread: stable-rate pose polling with timestamp."""
    global _pose_latest

    get_pose = mavc.get_pose
    sleep_time = 1.0 / _pose_thread_hz
    next_time = time.perf_counter()

    while not _stop_event.is_set():
        next_time += sleep_time

        # Get pose
        x, y, z, yaw, pitch, roll = get_pose()
        t = time.time()

        # Update shared pose
        _pose_latest = (x, y, z, yaw, pitch, roll)

        # Signal that pose is ready (only matters first time)
        _pose_ready.set()

        # Maintain stable loop timing
        sleep = next_time - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_time = time.perf_counter()


def start_pose_thread(frequency_hz):
    """Start pose thread at given frequency."""
    global _pose_thread, _pose_thread_hz

    if _pose_thread and _pose_thread.is_alive():
        print("[INFO] Pose thread already running")
        return

    _pose_thread_hz = frequency_hz
    _stop_event.clear()
    _pose_ready.clear()

    _pose_thread = threading.Thread(target=_pose_thread_worker, daemon=True)
    _pose_thread.start()

    print(f"[INFO] Pose thread started ({frequency_hz} Hz)")


def get_latest_pose():
    """
    Blocks until first pose is available, then returns immediately.
    Returns (t, x, y, z, yaw, pitch, roll)
    """
    _pose_ready.wait()   # efficient wait (no CPU burn)
    return _pose_latest


def stop_pose_thread():
    """Stop pose thread cleanly."""
    global _pose_thread

    _stop_event.set()

    if _pose_thread and _pose_thread.is_alive():
        _pose_thread.join(timeout=1.0)

    print("[INFO] Pose thread stopped")

def send_esp_cam_commands(ip, vflip, hflip, res_idx):
    import requests
    print("Sendingcommands to ESP32 CAM")
    def send(var, val):
        url = f"{ip}/control?var={var}&val={val}"
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"{var} set to {val}")
        except Exception as e:
            print(f"Error setting {var}: {e}")

    # Send commands
    send("vflip", vflip)
    send("hmirror", hflip)
    send("framesize", res_idx)
