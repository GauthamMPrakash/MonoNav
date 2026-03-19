import numpy as np

from utils.dstar_lite import DStarLitePlanner2D, select_lookahead_waypoint


def test_dstar_lite_finds_straight_path_on_empty_grid():
    planner = DStarLitePlanner2D(
        resolution=1.0,
        obstacle_buffer_m=0.0,
        bounds_padding_m=1.0,
        min_window_size_m=6.0,
        unknown_travel_cost=1.0,
    )

    result = planner.plan(np.array([0.0, 0.0]), np.array([4.0, 0.0]))

    assert result.found
    assert result.grid_path[0] == result.start_cell
    assert result.grid_path[-1] == result.goal_cell
    rows = {row for _, row in result.grid_path}
    assert len(rows) == 1


def test_incremental_replan_avoids_new_obstacle_without_rebuild():
    planner = DStarLitePlanner2D(
        resolution=1.0,
        obstacle_buffer_m=0.0,
        bounds_padding_m=1.0,
        min_window_size_m=8.0,
        unknown_travel_cost=1.0,
    )

    first = planner.plan(np.array([0.0, 0.0]), np.array([6.0, 0.0]))
    assert first.found

    middle_index = len(first.grid_path) // 2
    blocked_cell = first.grid_path[middle_index]
    blocked_world = planner.grid_spec.cell_to_world(blocked_cell).reshape(1, 2)

    second = planner.plan(np.array([0.0, 0.0]), np.array([6.0, 0.0]), obstacle_points_xy=blocked_world)

    assert second.found
    assert second.rebuilt is False
    assert blocked_cell not in second.grid_path


def test_blocked_goal_snaps_to_nearest_traversable_cell():
    planner = DStarLitePlanner2D(
        resolution=1.0,
        obstacle_buffer_m=0.0,
        bounds_padding_m=1.0,
        min_window_size_m=6.0,
        unknown_travel_cost=1.0,
    )

    nominal = planner.plan(np.array([0.0, 0.0]), np.array([4.0, 0.0]))
    assert nominal.found

    blocked_goal_world = planner.grid_spec.cell_to_world(nominal.goal_cell).reshape(1, 2)
    replanned = planner.plan(np.array([0.0, 0.0]), np.array([4.0, 0.0]), obstacle_points_xy=blocked_goal_world)

    assert replanned.found
    assert replanned.goal_cell != nominal.goal_cell


def test_select_lookahead_waypoint_advances_along_path():
    path = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.50, 0.0],
            [1.00, 0.0],
        ],
        dtype=float,
    )

    waypoint = select_lookahead_waypoint(
        world_path=path,
        current_xy=np.array([0.0, 0.0]),
        lookahead_m=0.4,
        reached_radius_m=0.1,
    )

    assert np.allclose(waypoint, np.array([0.5, 0.0]))
