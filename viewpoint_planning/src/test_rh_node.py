#!/usr/bin/env python3
""" K=10, H=3, step=0.12, lambda=2.0"""
import rospy
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from viewpoint_planners.viewpoint_planning import ViewpointPlanning
from metrics import compute_all_metrics, detect_occlusion_type, save_and_print
from plots.plot_coverage import plot_coverage_progression
from plots.plot_trajectory_3d import plot_3d_trajectory
from plots.plot_candidate_sequences import (
    plot_candidate_sequences,
    plot_candidate_sequences_grid,
)
from plots.plot_reconstruction import plot_reconstruction_comparison


if __name__ == "__main__":
    rospy.init_node("rh_test")
    NUM_ITERS = 12
    K, H = 10, 3

    occ = detect_occlusion_type()
    print(f"\n{'='*50}")
    print(f"RH Planner | K={K}, H={H} | Occlusion: {occ}")
    print(f"{'='*50}\n")

    vp = ViewpointPlanning(0)
    for i in range(NUM_ITERS):
        print(f"--- RH Iteration {i+1}/{NUM_ITERS} ---")
        vp.run_rh()

    target_voxels = vp.rh_planner.target_voxels
    mesh_coords = vp.rh_planner.mesh_coordinates
    if isinstance(target_voxels, np.ndarray) and target_voxels.ndim < 2:
        target_voxels = None

    ray_calls = vp.ray_calls_rh.tolist()

    results = compute_all_metrics(
        coverages=vp.coverages_rh.tolist(),
        recalls=vp.recall_rh.tolist(),
        precisions=vp.precision_rh.tolist(),
        distances=vp.trajectory_distance_rh.tolist(),
        times=vp.cumulative_time_rh.tolist(),
        ray_calls=ray_calls,
        method_name="RH-NBV",
        occlusion_type=occ,
        params={"K": K, "H": H, "step_size": 0.12, "lambda": 2.0},
        target_voxels=target_voxels,
        mesh_coordinates=mesh_coords,
    )
    results["sigma_series"] = vp.sigma_rh.tolist()
    results["occluded_recall_series"] = vp.occluded_recall_rh.tolist()
    save_and_print(results)

    os.makedirs("results", exist_ok=True)

    # Plot 1: Coverage progression
    try:
        plot_coverage_progression(
            coverages={"RH-NBV": vp.coverages_rh},
            save_path=f"results/coverage_rh_{occ}.png",
            title=f"RH-NBV Coverage Progression (Occlusion: {occ})",
        )
    except Exception as e:
        print(f"Coverage plot failed: {e}")

    # Plot 2: 3D Trajectory
    try:
        plot_3d_trajectory(
            trail=vp.trail_rh,
            mesh_coordinates=vp.rh_planner.mesh_coordinates,
            occlusion_type=occ,
            save_path=f"results/trajectory_3d_rh_{occ}.png",
            title=f"RH-NBV 3D Trajectory (Occlusion: {occ})",
            method_label='RH-NBV',
        )
    except Exception as e:
        print(f"Trajectory plot failed: {e}")

    # Plot 3 & 4: Candidate sequences
    try:
        if hasattr(vp.rh_planner, 'candidate_history') and len(vp.rh_planner.candidate_history) > 0:
            plot_candidate_sequences(
                candidate_history=vp.rh_planner.candidate_history,
                mesh_coordinates=vp.rh_planner.mesh_coordinates,
                occlusion_type=occ,
                iteration_to_plot=min(2, len(vp.rh_planner.candidate_history) - 1),
                save_path=f"results/candidates_iter2_{occ}.png",
            )
            plot_candidate_sequences_grid(
                candidate_history=vp.rh_planner.candidate_history,
                mesh_coordinates=vp.rh_planner.mesh_coordinates,
                occlusion_type=occ,
                iterations_to_plot=[i for i in [0, 3, 6, 9] if i < len(vp.rh_planner.candidate_history)],
                save_path=f"results/candidates_grid_{occ}.png",
            )
        else:
            print("Skipping candidate plots (no candidate_history attribute)")
    except Exception as e:
        print(f"Candidate plots failed: {e}")

    # Plot 5: Reconstruction comparison
    try:
        plot_reconstruction_comparison(
            target_voxels=vp.rh_planner.target_voxels,
            mesh_coordinates=vp.rh_planner.mesh_coordinates,
            save_path=f"results/reconstruction_rh_{occ}.png",
            method_label='RH-NBV',
        )
    except Exception as e:
        print(f"Reconstruction plot failed: {e}")

    print("\nAll plots done")
