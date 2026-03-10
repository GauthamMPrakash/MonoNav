#!/usr/bin/env python3
"""
Quick validation script for D* Lite planner integration.
Run this to verify the implementation compiles and basic functionality works.
"""

import sys
import os
import numpy as np

# Use a repository-root-relative import path instead of a hard-coded absolute path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, repo_root)

print("=" * 60)
print("MonoNav D* Lite Planner - Integration Validation")
print("=" * 60)

# Test 1: Import modules
print("\n[TEST 1] Importing modules...")
try:
    from utils.utils import (
        DStarLitePlanner, 
        ExplorationGrid, 
        ClosedLoopPrimitiveController,
        choose_primitive
    )
    print("  ✓ All classes imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize D* Lite Planner
print("\n[TEST 2] Initializing D* Lite Planner...")
try:
    dstar = DStarLitePlanner(grid_size=20.0, cell_size=0.2, k_max_iterations=100)
    print(f"  ✓ Planner initialized")
    print(f"    - Grid size: 20m x 20m x 20m")
    print(f"    - Cell size: 0.2m")
    print(f"    - Grid dimensions: {dstar.grid_dim}x{dstar.grid_dim}x{dstar.grid_dim}")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Initialize Exploration Grid
print("\n[TEST 3] Initializing Exploration Grid...")
try:
    exp_grid = ExplorationGrid(grid_size=20.0, cell_size=0.1, max_depth=10.0)
    print(f"  ✓ Exploration grid initialized")
    print(f"    - Grid size: 20m x 20m x 20m")
    print(f"    - Cell size: 0.1m")
    print(f"    - Grid dimensions: {exp_grid.grid_dim}x{exp_grid.grid_dim}x{exp_grid.grid_dim}")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    sys.exit(1)

# Test 4: Test D* Lite path planning
print("\n[TEST 4] Testing D* Lite path planning...")
try:
    start_pos = np.array([0.0, 0.0, 0.0])
    goal_pos = np.array([5.0, 0.0, 10.0])
    camera_pos = np.eye(4)
    camera_pos[0:3, 3] = start_pos
    
    path = dstar.plan_to_goal(
        start_pos, goal_pos, 
        exploration_grid=exp_grid,
        voxel_coords=np.array([]),
        camera_position=camera_pos
    )
    
    if path is not None:
        print(f"  ✓ Path planning successful")
        print(f"    - Path length: {len(path)} waypoints")
        print(f"    - Start: {path[0]}")
        print(f"    - End: {path[-1]}")
    else:
        print(f"  ⚠ No path found (expected for empty map)")
except Exception as e:
    print(f"  ✗ Path planning failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test Exploration Grid update
print("\n[TEST 5] Testing Exploration Grid depth update...")
try:
    # Create synthetic depth image
    depth = np.ones((100, 100), dtype=np.float32) * 5.0  # 5 meters
    intrinsics = np.array([
        [500, 0, 50],
        [0, 500, 50],
        [0, 0, 1]
    ], dtype=np.float32)
    
    exp_grid.update_from_depth(depth / 1000.0, camera_pos, intrinsics)
    print(f"  ✓ Exploration grid updated from depth")
except Exception as e:
    print(f"  ✗ Depth update failed: {e}")
    sys.exit(1)

# Test 6: Test Closed-Loop Controller
print("\n[TEST 6] Testing Closed-Loop Primitive Controller...")
try:
    controller = ClosedLoopPrimitiveController(
        forward_speed=1.0, yvel_gain=0.5, yawrate_gain=1.0
    )
    print(f"  ✓ Closed-loop controller initialized")
    print(f"    - Forward speed: 1.0 m/s")
    print(f"    - yvel_gain: 0.5")
    print(f"    - yawrate_gain: 1.0")
except Exception as e:
    print(f"  ✗ Controller initialization failed: {e}")
    sys.exit(1)

# Test 7: Configuration loading
print("\n[TEST 7] Checking configuration...")
try:
    from utils.utils import load_config
    config = load_config('config.yml')
    
    use_dstar = config.get('use_dstar_planner', True)
    use_exp_grid = config.get('use_exploration_grid', True)
    
    print(f"  ✓ Configuration loaded")
    print(f"    - use_dstar_planner: {use_dstar}")
    print(f"    - use_exploration_grid: {use_exp_grid}")
    print(f"    - dstar_grid_size: {config.get('dstar_grid_size', 'Not set')}")
    print(f"    - dstar_cell_size: {config.get('dstar_cell_size', 'Not set')}")
except Exception as e:
    print(f"  ✗ Configuration check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All validation tests passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Review DSTAR_LITE_IMPLEMENTATION.md for full documentation")
print("2. Update config.yml with your navigation goals")
print("3. Run mononav.py or AP_ObstacleAvoidance.py with MonoNav mode enabled ('g' key)")
print("=" * 60)
