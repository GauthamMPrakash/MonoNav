"""
A small utility that captures a sequence of depth frames from the
configured video stream (URL found in ``config.yml``) and computes a
few temporal-consistency statistics.

Only the number of frames need to be specified. Use ``--no-transform``
if you wish to skip the undistort/warp step (otherwise frames are
transformed using calibration parameters from config).

Example:

    python tests/depth_temporal_consistency.py --n-frames 100

The script reports mean/median/std/max/min of absolute differences
between consecutive frames and per-pixel variance, along with a simple
consistency score.
"""

import argparse
import glob
import os
import sys
import time
import numpy as np
import cv2
    
# ensure project root is on sys.path so we can import utils and config modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils.utils import compute_depth, transform_image, VideoCapture

def load_depth_frames(directory: str, ext: str = ".npy") -> np.ndarray:
    """Load all depth frames from ``directory`` with the given extension.
    Frames are sorted alphabetically before loading and stacked into a
    single array of shape (T, H, W).
    """
    pattern = os.path.join(directory, "*" + ext)
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise ValueError(f"no files found in {directory} matching *{ext}")

    frames = []
    for f in files:
        try:
            frames.append(np.load(f))
        except Exception as e:
            # try reading as an image if numpy fails
            import cv2

            depth = cv2.imread(f, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise
            frames.append(depth.astype(np.float32))

    return np.stack(frames)


def capture_and_estimate(n: int, source, model, size: int, transform: bool = True, mtx=None, dist=None, optimal_mtx=None, roi=None) -> np.ndarray:
    """Capture ``n`` RGB frames from ``source`` and estimate metric depth.

    The camera/stream is opened with OpenCV. Each colour frame is optionally
    undistorted and warped using ``transform_image`` (from ``utils.utils``)
    if ``transform`` is ``True`` and calibration parameters are supplied.
    The frame is then fed to ``compute_depth`` along with ``model`` and
    ``size``. The returned depth maps (converted to metres) are collected
    and displayed live as a colourised depth map.
    """

    cap = VideoCapture(source)
    if not cap:
        raise RuntimeError(f"unable to open source {source}")

    depths = []
    print("capturing", n, "frames; press q to abort early")
    while len(depths) < n:
        frame = cap.read()
        if frame is None:
            break
        if transform and mtx is not None and dist is not None and optimal_mtx is not None and roi is not None:
            frame = transform_image(frame, mtx, dist, optimal_mtx, roi)
        depth_mm, depth_colormap = compute_depth(frame, model, size)
        cv2.imshow("depth", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        depth_m = depth_mm.astype(np.float32) / 1000.0
        depths.append(depth_m)

    cap.cap.release()
    cv2.destroyAllWindows()
    return np.stack(depths)


def compute_stats(frames: np.ndarray) -> dict:
    """Compute temporal statistics on a stack of depth frames.

    ``frames`` has shape (T, H, W) and is expected to be in metres. All
    returned statistics therefore remain in metres.

    Metrics produced:

    * ``mean_abs_diff`` – the average absolute change between consecutive
      depth maps across all pixels and time steps. Low values indicate
      temporal stability.
    * ``median_abs_diff`` – median of absolute frame-to-frame changes;
      more robust to occasional spikes or outliers.
    * ``std_abs_diff`` – standard deviation of the absolute differences,
      giving a sense of variability in motion/noise over time.
    * ``max_abs_diff`` / ``min_abs_diff`` – extreme frame-to-frame
      changes (e.g., sudden jumps or completely static regions).
    * ``mean_variance`` – average over all pixels of the variance of their
      depth values across time; measures overall noise or drift.
    * ``median_variance`` – median per-pixel variance for robustness to
      a few highly noisy pixels.
    * ``max_variance`` / ``min_variance`` – pixels with the most and
      least temporal fluctuation.
    * ``consistency_score`` – a simple score defined as
      ``1 / (1 + mean_abs_diff)`` that lies in (0,1]; values closer to
      1 indicate very consistent frames.
    """
    if frames.ndim != 3:
        raise ValueError("frames must be a 3D array (T,H,W)")

    diffs = np.abs(np.diff(frames, axis=0))
    stats = {
        "mean_abs_diff": float(diffs.mean()),
        "median_abs_diff": float(np.median(diffs)),
        "std_abs_diff": float(diffs.std()),
        "max_abs_diff": float(diffs.max()),
        "min_abs_diff": float(diffs.min()),
    }

    # per-pixel variance across time
    var = frames.var(axis=0)
    stats.update(
        {
            "mean_variance": float(var.mean()),
            "median_variance": float(np.median(var)),
            "max_variance": float(var.max()),
            "min_variance": float(var.min()),
        }
    )

    # simple temporal consistency score: 1 / (1 + mean_abs_diff)
    stats["consistency_score"] = float(1.0 / (1.0 + stats["mean_abs_diff"]))

    return stats


if __name__ == "__main__":
    # read camera stream URL and depth parameters from config
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for this script; install it with 'pip install pyyaml'") from e
    import torch
    # ensure DepthAnythingV2 package is on sys.path (same as other scripts)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metric_path = os.path.join(repo_root, "DepthAnythingV2-metric")
    if metric_path not in sys.path:
        sys.path.insert(0, metric_path)
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as e:
        raise ImportError(
            "could not import DepthAnythingV2; ensure 'DepthAnythingV2-metric' "
            "directory is present and the package has been installed or added to PYTHONPATH"
        ) from e

    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    source = cfg.get("camera_ip", 0)

    # model settings pulled from config
    INPUT_SIZE = cfg["INPUT_SIZE"]
    CHECKPOINT = "../"+cfg["DA2_CHECKPOINT"]
    ENCODER = CHECKPOINT[-8:-4]
    if ENCODER is None:
        ENCODER = CHECKPOINT.split("_")[-1].split(".")[0]
        print(f"[warning] DA2_ENCODER not set; parsed '{ENCODER}' from checkpoint")
    MAX_DEPTH = cfg.get("DEPTH_MAX", 20)
    # calibration values for transforming raw frames to undistorted crop
    camera_calibration_path = cfg.get("camera_calibration_path")
    if camera_calibration_path:
        try:
            from utils.utils import get_calibration_values

            mtx, dist, optimal_mtx, roi = get_calibration_values(camera_calibration_path)
        except Exception as e:
            print(f"[warning] failed to load calibration from {camera_calibration_path}: {e}")
            mtx = dist = optimal_mtx = roi = None
    else:
        mtx = dist = optimal_mtx = roi = None
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # same model config dictionary as mononav
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    depth_model = DepthAnythingV2(**{**model_configs[ENCODER], "max_depth": MAX_DEPTH})
    depth_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    depth_model = depth_model.to(DEVICE).eval()

    parser = argparse.ArgumentParser(
        description="Capture RGB frames from the configured stream, estimate depth live, and compute temporal consistency."
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="number of frames to capture from the stream",
    )
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="do not apply camera undistort/warp to frames before depth estimation",
    )
    args = parser.parse_args()

    # try converting camera_ip to integer if possible
    try:
        source = int(source)
    except Exception:
        pass
    init = time.time()
    frames = capture_and_estimate(
        args.n_frames,
        source,
        depth_model,
        INPUT_SIZE,
        transform=not args.no_transform,
        mtx=mtx,
        dist=dist,
        optimal_mtx=optimal_mtx,
        roi=roi,
    )
    print(f"captured {frames.shape[0]} depth frames of shape {frames.shape[1:]}\n")
    stats = compute_stats(frames)
    for k, v in stats.items():
        print(f"{k}: {v}")
    end = time.time()
    print(f"\nTotal time: {end - init:.4f} seconds")
    print(f"{args.n_frames/(end - init):.2f} frames per second")

