#!/usr/bin/env python3
"""
test_rh_node_box.py
-------------------
Sanity-check experiment: RH planner on a simple 15 cm box target.

Use this instead of test_rh_node.py when you want to verify the full
pipeline (arm motion, voxel update, metrics, plots) without the
complexity of the Stanford Bunny mesh.

Expected sanity checks:
  - coverage > 0 after iteration 1  (box is easy to see)
  - coverage > 50% within 4–5 iters (simple geometry)
  - F1 > 0.5 within 6 iters
  - No arm failures for no-occlusion scenario
"""

import rospy
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")

from viewpoint_planners.viewpoint_planning_box import ViewpointPlanningBox
from metrics import compute_all_metrics, detect_occlusion_type, save_and_print
from plots.plot_coverage import plot_coverage_progression
from plots.plot_trajectory_3d import plot_3d_trajectory
from plots.plot_reconstruction import plot_reconstruction_comparison
import subprocess

if __name__ == "__main__":
    rospy.init_node("rh_test_box")
    # Spawn box at correct position via Gazebo service
    subprocess.run([
        "rosrun", "gazebo_ros", "spawn_model",
        "-file", "/home/ayse/catkin_ws/src/PSO-for-perception-of-occluded-target-nodes/simulation_environment/worlds/box.world",
        "-sdf", "-model", "target_box",
        "-x", "0.5", "-y", "-0.4", "-z", "1.1"
], check=False)

    NUM_ITERS = 12
    K, H = 10, 3

    # detect_occlusion_type reads viewpoint_planning.py — override manually
    # for the box file since they live in separate files.
    occ = detect_occlusion_type(
        config_path="viewpoint_planners/viewpoint_planning_box.py"
    )
    print(f"\n{'='*55}")
    print(f"RH-Box Planner | K={K}, H={H} | Occlusion: {occ}")
    print(f"Target: simple 15 cm box  (sanity-check experiment)")
    print(f"{'='*55}\n")

    vp = ViewpointPlanningBox(lr=0)

    for i in range(NUM_ITERS):
        print(f"--- Box Iteration {i+1}/{NUM_ITERS} ---")
        coverage, loss, f1, recall, precision, n_evals = vp.run_rh()

        # ----------------------------------------------------------------
        # Sanity checks — fail loudly so problems are obvious early
        # ----------------------------------------------------------------
        if i == 0 and coverage == 0.0:
            print(
                "[WARN] Coverage is 0 after first iteration.\n"
                "       Check: (1) box spawned in Gazebo, "
                "(2) color segmentation sees red, "
                "(3) camera pose is within reach bounds."
            )
        if i == 2 and f1 == 0.0:
            print(
                "[WARN] F1 is still 0 after 3 iterations.\n"
                "       Check: mesh_coordinates match Gazebo box pose."
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    target_voxels = vp.rh_planner.target_voxels
    mesh_coords   = vp.rh_planner.mesh_coordinates
    if isinstance(target_voxels, np.ndarray) and target_voxels.ndim < 2:
        target_voxels = None

    results = compute_all_metrics(
        coverages=vp.coverages_rh.tolist(),
        recalls=vp.recall_rh.tolist(),
        precisions=vp.precision_rh.tolist(),
        distances=vp.trajectory_distance_rh.tolist(),
        times=vp.cumulative_time_rh.tolist(),
        ray_calls=vp.ray_calls_rh.tolist(),
        method_name="RH-NBV-Box",
        occlusion_type=occ,
        params={"K": K, "H": H, "step_size": 0.065, "lambda": 2.0,
                "target": "box_15cm"},
        target_voxels=target_voxels,
        mesh_coordinates=mesh_coords,
    )
    results["sigma_series"]          = vp.sigma_rh.tolist()
    results["occluded_recall_series"] = vp.occluded_recall_rh.tolist()
    save_and_print(results, prefix="results/box")

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        plot_coverage_progression(
            coverages={"RH-NBV-Box": vp.coverages_rh},
            save_path=f"results/coverage_box_{occ}.png",
            title=f"RH-NBV Box Coverage (Occlusion: {occ})",
        )
        print("Coverage plot saved.")
    except Exception as e:
        print(f"Coverage plot failed: {e}")

    try:
        plot_3d_trajectory(
            trail=vp.trail_rh,
            mesh_coordinates=vp.rh_planner.mesh_coordinates,
            occlusion_type=occ,
            save_path=f"results/trajectory_3d_box_{occ}.png",
            title=f"RH-NBV Box 3D Trajectory (Occlusion: {occ})",
            method_label="RH-NBV-Box",
        )
        print("Trajectory plot saved.")
    except Exception as e:
        print(f"Trajectory plot failed: {e}")

    try:
        plot_reconstruction_comparison(
            target_voxels=vp.rh_planner.target_voxels,
            mesh_coordinates=vp.rh_planner.mesh_coordinates,
            save_path=f"results/reconstruction_box_{occ}.png",
            method_label="RH-NBV-Box",
        )
        print("Reconstruction plot saved.")
    except Exception as e:
        print(f"Reconstruction plot failed: {e}")

    print("\nAll done.")
