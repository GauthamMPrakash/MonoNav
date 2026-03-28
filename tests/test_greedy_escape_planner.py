import unittest

import numpy as np

from utils.utils import GreedyEscapePlanner, GreedyEscapePlannerParams


class _FakeVBG:
    def __init__(self, voxel_coords_xyz: np.ndarray):
        voxel_coords_xyz = np.asarray(voxel_coords_xyz, dtype=np.float64)
        self._coords = voxel_coords_xyz
        self._weights = np.ones((voxel_coords_xyz.shape[0],), dtype=np.float64)
        self._tsdf = -np.ones((voxel_coords_xyz.shape[0],), dtype=np.float64)

    def attribute(self, name: str):
        if name == "weight":
            return self._weights
        if name == "tsdf":
            return self._tsdf
        raise KeyError(name)

    def voxel_coordinates_and_flattened_indices(self):
        idx = np.arange(int(self._coords.shape[0]), dtype=np.int64)
        return self._coords, idx


def _make_traj_linesets():
    class _Traj:
        def __init__(self, points: np.ndarray):
            self.points = np.asarray(points, dtype=np.float64)

        def transform(self, T: np.ndarray):
            pts = self.points
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
            out = (T @ pts_h.T).T[:, :3]
            self.points = out
            return self

    def lineset_for_x(end_x: float):
        pts = np.array([[0.0, 0.0, 0.0], [end_x, 0.0, 1.0]], dtype=np.float64)
        return _Traj(pts)

    return [lineset_for_x(-1.0), lineset_for_x(0.0), lineset_for_x(1.0)]


class TestGreedyEscapePlanner(unittest.TestCase):
    def test_enters_escape_after_stagnation(self):
        vbg = _FakeVBG(np.array([[0.2, 0.0, 0.5]], dtype=np.float32))
        traj_linesets = _make_traj_linesets()
        goal = np.array([[-5.0, 0.0, 1.0]], dtype=np.float64)
        cam = np.eye(4, dtype=np.float64)

        planner = GreedyEscapePlanner(GreedyEscapePlannerParams(enabled=True, progress_eps_m=0.0, stagnation_steps=1, escape_min_steps=1))
        self.assertEqual(planner.mode, "goal")

        _ = planner.choose(vbg, cam, traj_linesets, goal, 0.1, False, False, False, 0.0)
        self.assertEqual(planner.mode, "goal")

        planner.params.escape_min_steps = 10
        _ = planner.choose(vbg, cam, traj_linesets, goal, 0.1, False, False, False, 0.0)
        self.assertEqual(planner.mode, "escape")

    def test_escape_prefers_goal_side_with_clearance(self):
        # Obstacle voxels are placed near the straight trajectory (x≈0.2), making left safer.
        vbg = _FakeVBG(np.array([[0.2, 0.0, 0.5], [0.2, 0.0, 0.8]], dtype=np.float32))
        traj_linesets = _make_traj_linesets()
        goal_left = np.array([[-5.0, 0.0, 1.0]], dtype=np.float64)
        cam = np.eye(4, dtype=np.float64)

        planner = GreedyEscapePlanner(GreedyEscapePlannerParams(enabled=True, progress_eps_m=0.0, stagnation_steps=1, escape_min_steps=10))
        _ = planner.choose(vbg, cam, traj_linesets, goal_left, 0.1, False, False, False, 0.0)
        chosen = planner.choose(vbg, cam, traj_linesets, goal_left, 0.1, False, False, False, 0.0)

        # Expect escape mode and the left trajectory (index 0) due to higher clearance.
        self.assertEqual(planner.mode, "escape")
        self.assertEqual(chosen, 0)


if __name__ == "__main__":
    unittest.main()
