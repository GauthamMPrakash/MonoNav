from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Optional

import numpy as np


INF = float("inf")


GridCell = tuple[int, int]


@dataclass(frozen=True)
class GridSpec2D:
    origin_xy: np.ndarray
    resolution: float
    width: int
    height: int

    def in_bounds(self, cell: GridCell) -> bool:
        col, row = cell
        return 0 <= col < self.width and 0 <= row < self.height

    def world_to_cell(self, point_xy: np.ndarray, clamp: bool = False) -> Optional[GridCell]:
        point_xy = np.asarray(point_xy, dtype=float)
        rel = (point_xy - self.origin_xy) / self.resolution
        col = int(math.floor(rel[0]))
        row = int(math.floor(rel[1]))
        if clamp:
            col = min(max(col, 0), self.width - 1)
            row = min(max(row, 0), self.height - 1)
            return (col, row)
        cell = (col, row)
        if self.in_bounds(cell):
            return cell
        return None

    def cell_to_world(self, cell: GridCell) -> np.ndarray:
        col, row = cell
        return self.origin_xy + self.resolution * np.array([col + 0.5, row + 0.5], dtype=float)


@dataclass
class PathPlanResult:
    world_path: np.ndarray
    grid_path: list[GridCell]
    start_cell: Optional[GridCell]
    goal_cell: Optional[GridCell]
    rebuilt: bool
    changed_cells: int
    reason: str = ""

    @property
    def found(self) -> bool:
        return len(self.grid_path) > 0


class DStarLite:
    def __init__(
        self,
        grid_spec: GridSpec2D,
        occupancy: np.ndarray,
        observed: np.ndarray,
        start: GridCell,
        goal: GridCell,
        unknown_travel_cost: float = 2.5,
    ):
        self.grid_spec = grid_spec
        self.occupancy = occupancy.astype(bool, copy=True)
        self.observed = observed.astype(bool, copy=True)
        self.unknown_travel_cost = max(1.0, float(unknown_travel_cost))
        self.start = start
        self.last_start = start
        self.goal = goal
        self.km = 0.0

        self.g = np.full((grid_spec.height, grid_spec.width), INF, dtype=float)
        self.rhs = np.full((grid_spec.height, grid_spec.width), INF, dtype=float)
        self.rhs[goal[1], goal[0]] = 0.0

        self._heap: list[tuple[float, float, int, GridCell]] = []
        self._open_keys: dict[GridCell, tuple[float, float]] = {}
        self._counter = 0
        self._push(goal, self.calculate_key(goal))

    def calculate_key(self, cell: GridCell) -> tuple[float, float]:
        g_rhs = min(self._g(cell), self._rhs(cell))
        return (g_rhs + self._heuristic(self.start, cell) + self.km, g_rhs)

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        diag = min(dx, dy)
        straight = max(dx, dy) - diag
        return math.sqrt(2.0) * diag + straight

    def _g(self, cell: GridCell) -> float:
        return self.g[cell[1], cell[0]]

    def _rhs(self, cell: GridCell) -> float:
        return self.rhs[cell[1], cell[0]]

    def _set_g(self, cell: GridCell, value: float) -> None:
        self.g[cell[1], cell[0]] = value

    def _set_rhs(self, cell: GridCell, value: float) -> None:
        self.rhs[cell[1], cell[0]] = value

    def _push(self, cell: GridCell, key: tuple[float, float]) -> None:
        self._open_keys[cell] = key
        heapq.heappush(self._heap, (key[0], key[1], self._counter, cell))
        self._counter += 1

    def _discard(self, cell: GridCell) -> None:
        self._open_keys.pop(cell, None)

    def _top_key(self) -> tuple[float, float]:
        while self._heap:
            k1, k2, _, cell = self._heap[0]
            if self._open_keys.get(cell) != (k1, k2):
                heapq.heappop(self._heap)
                continue
            return (k1, k2)
        return (INF, INF)

    def _pop(self) -> tuple[GridCell, tuple[float, float]]:
        while self._heap:
            k1, k2, _, cell = heapq.heappop(self._heap)
            if self._open_keys.get(cell) != (k1, k2):
                continue
            self._open_keys.pop(cell, None)
            return cell, (k1, k2)
        raise KeyError("priority queue is empty")

    def _node_cost(self, cell: GridCell) -> float:
        if self.occupancy[cell[1], cell[0]]:
            return INF
        if self.observed[cell[1], cell[0]]:
            return 1.0
        return self.unknown_travel_cost

    def _edge_cost(self, src: GridCell, dst: GridCell) -> float:
        src_cost = self._node_cost(src)
        dst_cost = self._node_cost(dst)
        if not math.isfinite(src_cost) or not math.isfinite(dst_cost):
            return INF
        diagonal = src[0] != dst[0] and src[1] != dst[1]
        if diagonal:
            corner_a = (dst[0], src[1])
            corner_b = (src[0], dst[1])
            if self.occupancy[corner_a[1], corner_a[0]] or self.occupancy[corner_b[1], corner_b[0]]:
                return INF
        step_cost = math.sqrt(2.0) if diagonal else 1.0
        return step_cost * max(src_cost, dst_cost)

    def neighbors(self, cell: GridCell) -> Iterable[GridCell]:
        col, row = cell
        for dcol in (-1, 0, 1):
            for drow in (-1, 0, 1):
                if dcol == 0 and drow == 0:
                    continue
                nxt = (col + dcol, row + drow)
                if self.grid_spec.in_bounds(nxt):
                    yield nxt

    def predecessors(self, cell: GridCell) -> Iterable[GridCell]:
        return self.neighbors(cell)

    def update_vertex(self, cell: GridCell) -> None:
        if cell != self.goal:
            best_rhs = INF
            for neighbor in self.neighbors(cell):
                candidate = self._edge_cost(cell, neighbor) + self._g(neighbor)
                if candidate < best_rhs:
                    best_rhs = candidate
            self._set_rhs(cell, best_rhs)

        self._discard(cell)
        if not math.isclose(self._g(cell), self._rhs(cell), rel_tol=0.0, abs_tol=1e-9):
            self._push(cell, self.calculate_key(cell))

    def update_cell(self, cell: GridCell, occupied: bool, observed: bool) -> None:
        if not self.grid_spec.in_bounds(cell):
            return
        if self.occupancy[cell[1], cell[0]] == occupied and self.observed[cell[1], cell[0]] == observed:
            return
        self.occupancy[cell[1], cell[0]] = occupied
        self.observed[cell[1], cell[0]] = observed

        affected = {cell}
        affected.update(self.predecessors(cell))
        for node in affected:
            self.update_vertex(node)

    def move_start(self, new_start: GridCell) -> None:
        self.km += self._heuristic(self.last_start, new_start)
        self.last_start = new_start
        self.start = new_start

    def compute_shortest_path(self, max_iterations: Optional[int] = None) -> None:
        iterations = 0
        if max_iterations is None:
            max_iterations = self.grid_spec.width * self.grid_spec.height * 20

        while (
            self._top_key() < self.calculate_key(self.start)
            or not math.isclose(self._rhs(self.start), self._g(self.start), rel_tol=0.0, abs_tol=1e-9)
        ):
            if iterations >= max_iterations:
                break
            try:
                cell, old_key = self._pop()
            except KeyError:
                break
            new_key = self.calculate_key(cell)
            if old_key < new_key:
                self._push(cell, new_key)
            elif self._g(cell) > self._rhs(cell):
                self._set_g(cell, self._rhs(cell))
                for pred in self.predecessors(cell):
                    self.update_vertex(pred)
            else:
                self._set_g(cell, INF)
                self.update_vertex(cell)
                for pred in self.predecessors(cell):
                    self.update_vertex(pred)
            iterations += 1

    def extract_path(self, start: Optional[GridCell] = None, max_steps: Optional[int] = None) -> list[GridCell]:
        if start is None:
            start = self.start
        if max_steps is None:
            max_steps = self.grid_spec.width * self.grid_spec.height

        if not math.isfinite(self._g(start)) and not math.isfinite(self._rhs(start)):
            return []

        path = [start]
        current = start
        visited = {start}

        for _ in range(max_steps):
            if current == self.goal:
                return path
            best_neighbor = None
            best_score = INF
            for neighbor in self.neighbors(current):
                score = self._edge_cost(current, neighbor) + self._g(neighbor)
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor
            if best_neighbor is None or not math.isfinite(best_score):
                return []
            if best_neighbor in visited:
                return []
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor

        return []


class DStarLitePlanner2D:
    def __init__(
        self,
        resolution: float = 0.25,
        obstacle_buffer_m: float = 0.8,
        bounds_padding_m: float = 1.5,
        min_window_size_m: float = 8.0,
        unknown_travel_cost: float = 2.5,
    ):
        self.resolution = float(resolution)
        self.obstacle_buffer_m = max(0.0, float(obstacle_buffer_m))
        self.bounds_padding_m = max(self.resolution, float(bounds_padding_m))
        self.min_window_size_m = max(self.resolution, float(min_window_size_m))
        self.unknown_travel_cost = max(1.0, float(unknown_travel_cost))

        self.grid_spec: Optional[GridSpec2D] = None
        self.occupancy: Optional[np.ndarray] = None
        self.observed: Optional[np.ndarray] = None
        self.planner: Optional[DStarLite] = None
        self.goal_cell: Optional[GridCell] = None

    def plan(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        obstacle_points_xy: Optional[np.ndarray] = None,
        observed_points_xy: Optional[np.ndarray] = None,
    ) -> PathPlanResult:
        start_xy = np.asarray(start_xy, dtype=float).reshape(2)
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        obstacle_points_xy = _as_xy_array(obstacle_points_xy)
        observed_points_xy = _as_xy_array(observed_points_xy)

        grid_spec, rebuilt = self._ensure_grid_spec(start_xy, goal_xy, obstacle_points_xy, observed_points_xy)
        occupancy, observed = self._rasterize(grid_spec, obstacle_points_xy, observed_points_xy)

        raw_start = grid_spec.world_to_cell(start_xy, clamp=True)
        raw_goal = grid_spec.world_to_cell(goal_xy, clamp=True)
        start_cell = _nearest_traversable_cell(raw_start, occupancy)
        goal_cell = _nearest_traversable_cell(raw_goal, occupancy)

        if start_cell is None or goal_cell is None:
            self.grid_spec = grid_spec
            self.occupancy = occupancy
            self.observed = observed
            self.planner = None
            self.goal_cell = goal_cell
            return PathPlanResult(
                world_path=np.empty((0, 2), dtype=float),
                grid_path=[],
                start_cell=start_cell,
                goal_cell=goal_cell,
                rebuilt=rebuilt,
                changed_cells=0,
                reason="no traversable start or goal cell",
            )

        if rebuilt or self.planner is None or self.goal_cell != goal_cell:
            self.grid_spec = grid_spec
            self.occupancy = occupancy
            self.observed = observed
            self.goal_cell = goal_cell
            self.planner = DStarLite(
                grid_spec=grid_spec,
                occupancy=occupancy,
                observed=observed,
                start=start_cell,
                goal=goal_cell,
                unknown_travel_cost=self.unknown_travel_cost,
            )
            changed_cells = int(np.count_nonzero(occupancy) + np.count_nonzero(observed))
        else:
            changed_mask = (occupancy != self.occupancy) | (observed != self.observed)
            changed_indices = np.argwhere(changed_mask)
            changed_cells = int(len(changed_indices))
            self.occupancy[:, :] = occupancy
            self.observed[:, :] = observed
            for row, col in changed_indices:
                self.planner.update_cell((int(col), int(row)), bool(occupancy[row, col]), bool(observed[row, col]))
            self.planner.move_start(start_cell)

        self.planner.compute_shortest_path()
        grid_path = self.planner.extract_path(start_cell)
        world_path = (
            np.asarray([self.grid_spec.cell_to_world(cell) for cell in grid_path], dtype=float)
            if grid_path
            else np.empty((0, 2), dtype=float)
        )
        reason = "" if grid_path else "planner could not extract a path"

        return PathPlanResult(
            world_path=world_path,
            grid_path=grid_path,
            start_cell=start_cell,
            goal_cell=goal_cell,
            rebuilt=rebuilt,
            changed_cells=changed_cells,
            reason=reason,
        )

    def _ensure_grid_spec(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        obstacle_points_xy: np.ndarray,
        observed_points_xy: np.ndarray,
    ) -> tuple[GridSpec2D, bool]:
        if self.grid_spec is not None:
            points = [start_xy.reshape(1, 2), goal_xy.reshape(1, 2)]
            if len(obstacle_points_xy) > 0:
                points.append(obstacle_points_xy)
            if len(observed_points_xy) > 0:
                points.append(observed_points_xy)
            stacked = np.vstack(points)
            if _points_fit_grid(self.grid_spec, stacked, self.bounds_padding_m):
                return self.grid_spec, False

        points = [start_xy.reshape(1, 2), goal_xy.reshape(1, 2)]
        if len(obstacle_points_xy) > 0:
            points.append(obstacle_points_xy)
        if len(observed_points_xy) > 0:
            points.append(observed_points_xy)
        stacked = np.vstack(points)
        min_xy = np.min(stacked, axis=0) - self.bounds_padding_m
        max_xy = np.max(stacked, axis=0) + self.bounds_padding_m
        span = np.maximum(max_xy - min_xy, self.min_window_size_m)
        center = 0.5 * (min_xy + max_xy)
        min_xy = center - 0.5 * span
        width = int(math.ceil(span[0] / self.resolution)) + 1
        height = int(math.ceil(span[1] / self.resolution)) + 1
        grid_spec = GridSpec2D(
            origin_xy=min_xy.astype(float),
            resolution=self.resolution,
            width=max(width, 1),
            height=max(height, 1),
        )
        return grid_spec, True

    def _rasterize(
        self,
        grid_spec: GridSpec2D,
        obstacle_points_xy: np.ndarray,
        observed_points_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        occupancy = np.zeros((grid_spec.height, grid_spec.width), dtype=bool)
        observed = np.zeros((grid_spec.height, grid_spec.width), dtype=bool)

        if len(observed_points_xy) > 0:
            cols_rows = np.floor((observed_points_xy - grid_spec.origin_xy) / grid_spec.resolution).astype(int)
            mask = (
                (cols_rows[:, 0] >= 0)
                & (cols_rows[:, 0] < grid_spec.width)
                & (cols_rows[:, 1] >= 0)
                & (cols_rows[:, 1] < grid_spec.height)
            )
            cols_rows = cols_rows[mask]
            if len(cols_rows) > 0:
                observed[cols_rows[:, 1], cols_rows[:, 0]] = True

        if len(obstacle_points_xy) > 0:
            cols_rows = np.floor((obstacle_points_xy - grid_spec.origin_xy) / grid_spec.resolution).astype(int)
            mask = (
                (cols_rows[:, 0] >= 0)
                & (cols_rows[:, 0] < grid_spec.width)
                & (cols_rows[:, 1] >= 0)
                & (cols_rows[:, 1] < grid_spec.height)
            )
            cols_rows = cols_rows[mask]
            if len(cols_rows) > 0:
                occupancy[cols_rows[:, 1], cols_rows[:, 0]] = True

        inflated = _inflate_obstacles(occupancy, self.obstacle_buffer_m, grid_spec.resolution)
        observed |= inflated
        return inflated, observed


def select_lookahead_waypoint(
    world_path: np.ndarray,
    current_xy: np.ndarray,
    lookahead_m: float,
    reached_radius_m: float,
) -> Optional[np.ndarray]:
    if world_path is None or len(world_path) == 0:
        return None

    current_xy = np.asarray(current_xy, dtype=float).reshape(2)
    reached_radius_m = max(0.0, float(reached_radius_m))
    lookahead_m = max(0.0, float(lookahead_m))

    path = np.asarray(world_path, dtype=float)
    start_idx = 0
    while start_idx < len(path) - 1 and np.linalg.norm(path[start_idx] - current_xy) <= reached_radius_m:
        start_idx += 1

    target = path[start_idx]
    if lookahead_m == 0.0:
        return target

    accumulated = 0.0
    previous = current_xy
    for idx in range(start_idx, len(path)):
        step = float(np.linalg.norm(path[idx] - previous))
        accumulated += step
        target = path[idx]
        if accumulated >= lookahead_m:
            return target
        previous = path[idx]
    return target


def _as_xy_array(points_xy: Optional[np.ndarray]) -> np.ndarray:
    if points_xy is None:
        return np.empty((0, 2), dtype=float)
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=float)
    return points_xy.reshape((-1, 2))


def _points_fit_grid(grid_spec: GridSpec2D, points_xy: np.ndarray, margin_m: float) -> bool:
    min_bound = grid_spec.origin_xy + margin_m
    max_bound = grid_spec.origin_xy + grid_spec.resolution * np.array([grid_spec.width, grid_spec.height], dtype=float) - margin_m
    return bool(np.all(points_xy >= min_bound) and np.all(points_xy <= max_bound))


def _nearest_traversable_cell(cell: Optional[GridCell], occupancy: np.ndarray) -> Optional[GridCell]:
    if cell is None:
        return None
    col, row = cell
    height, width = occupancy.shape
    if 0 <= row < height and 0 <= col < width and not occupancy[row, col]:
        return cell

    visited = np.zeros_like(occupancy, dtype=bool)
    queue = deque([cell])
    while queue:
        col, row = queue.popleft()
        if not (0 <= row < height and 0 <= col < width):
            continue
        if visited[row, col]:
            continue
        visited[row, col] = True
        if not occupancy[row, col]:
            return (col, row)
        for dcol in (-1, 0, 1):
            for drow in (-1, 0, 1):
                if dcol == 0 and drow == 0:
                    continue
                queue.append((col + dcol, row + drow))
    return None


def _inflate_obstacles(occupancy: np.ndarray, inflation_radius_m: float, resolution: float) -> np.ndarray:
    if inflation_radius_m <= 0.0 or not np.any(occupancy):
        return occupancy.copy()

    inflation_cells = int(math.ceil(inflation_radius_m / resolution))
    offsets = []
    for dcol in range(-inflation_cells, inflation_cells + 1):
        for drow in range(-inflation_cells, inflation_cells + 1):
            if math.hypot(dcol, drow) * resolution <= inflation_radius_m + 1e-9:
                offsets.append((dcol, drow))

    inflated = occupancy.copy()
    rows, cols = np.nonzero(occupancy)
    height, width = occupancy.shape
    for row, col in zip(rows, cols):
        for dcol, drow in offsets:
            new_col = col + dcol
            new_row = row + drow
            if 0 <= new_row < height and 0 <= new_col < width:
                inflated[new_row, new_col] = True
    return inflated
