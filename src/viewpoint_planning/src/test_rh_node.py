#!/usr/bin/env python3
"""
test_rh_node.py — ROS 2 Jazzy / Burusa-aligned RH-NBV evaluation driver.

Usage:
    ./run_ros2_gz.sh                        # defaults: K=10 H=3, 1 trial
    NUM_TRIALS=4 ./run_ros2_gz.sh
    RH_K=20 RH_H=2 OCC=well ./run_ros2_gz.sh
    OCC=frontal EXPERIMENT=C ./run_ros2_gz.sh
"""
import os
import json
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ros2_node  # ROS 2 singleton node (replaces rospy.init_node)

# Absolute results root so ROS2's CWD changes don't break relative paths
_RESULTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
)

from viewpoint_planners.viewpoint_planning import ViewpointPlanning
from metrics import compute_all_metrics, detect_occlusion_type, save_and_print
from plots.plot_coverage import plot_coverage_progression
from plots.plot_trajectory_3d import plot_3d_trajectory
from plots.plot_candidate_sequences import (
    plot_candidate_sequences,
    plot_candidate_sequences_grid,
)
from plots.plot_reconstruction import (
    plot_reconstruction_comparison,
    plot_reconstruction_evolution_grid,
    plot_reconstruction_single_iter,
)

from viewpoint_planners.fair_comparison_config import BASE_SEED as FC_BASE_SEED

EXPERIMENT = os.environ.get("EXPERIMENT", "D").upper()
_default_iters = 20 if EXPERIMENT == "C" else 5
NUM_ITERS  = int(os.environ.get("NUM_ITERS", _default_iters))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", 1))
BASE_SEED  = FC_BASE_SEED

RH_PARAMS = {
    "horizon":        int(os.environ.get("RH_H", 3)),
    "num_candidates": int(os.environ.get("RH_K", 10)),
    "lambda_cost":    float(os.environ.get("RH_LAMBDA", 2.0)),
    "discount":       float(os.environ.get("RH_GAMMA", 0.85)),
    "step_size":      float(os.environ.get("RH_STEP", 0.065)),
    "use_spherical_bounds": os.environ.get("RH_SHELL", "0") != "0",
    "occlusion_bonus":      float(os.environ.get("OCC_BONUS", 2.0)),
    "stagnation_patience":  int(os.environ.get("RH_STAG_P", 4)),
    "stagnation_threshold": float(os.environ.get("RH_STAG_T", 1.5)),
}
K = RH_PARAMS["num_candidates"]
H = RH_PARAMS["horizon"]


def make_run_dir(occ):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shell = "shell" if RH_PARAMS["use_spherical_bounds"] else "box"
    name = f"run_{ts}_exp{EXPERIMENT}_{occ}_K{K}_H{H}_{shell}"
    run_dir = os.path.join(_RESULTS_ROOT, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_plots(vp, occ, out_dir, recon_snapshots=None):
    rh = vp.rh_planner

    def _try(fn, label):
        try:
            fn()
        except Exception as e:
            print(f"  [plot] {label} failed: {e}")

    _try(lambda: plot_coverage_progression(
        coverages={"RH-NBV": vp.coverages_rh},
        save_path=os.path.join(out_dir, f"coverage_rh_{occ}.png"),
        title=f"RH-NBV Coverage (Occlusion: {occ}, K={K}, H={H})",
    ), "coverage")

    _try(lambda: plot_3d_trajectory(
        trail=vp.trail_rh,
        mesh_coordinates=rh.mesh_coordinates,
        occlusion_type=occ,
        save_path=os.path.join(out_dir, f"trajectory_3d_rh_{occ}.png"),
        title=f"RH-NBV 3D Trajectory (Occlusion: {occ})",
        method_label="RH-NBV",
    ), "trajectory")

    if hasattr(rh, "candidate_history") and len(rh.candidate_history) > 0:
        _try(lambda: plot_candidate_sequences(
            candidate_history=rh.candidate_history,
            mesh_coordinates=rh.mesh_coordinates,
            occlusion_type=occ,
            iteration_to_plot=min(2, len(rh.candidate_history) - 1),
            save_path=os.path.join(out_dir, f"candidates_iter2_{occ}.png"),
        ), "candidates_single")

        _try(lambda: plot_candidate_sequences_grid(
            candidate_history=rh.candidate_history,
            mesh_coordinates=rh.mesh_coordinates,
            occlusion_type=occ,
            iterations_to_plot=[i for i in [0, 1, 2, 3] if i < len(rh.candidate_history)],
            save_path=os.path.join(out_dir, f"candidates_grid_{occ}.png"),
        ), "candidates_grid")
    else:
        print("  [plot] skipping candidate plots (no candidate_history)")

    _try(lambda: plot_reconstruction_comparison(
        target_voxels=rh.target_voxels,
        mesh_coordinates=rh.mesh_coordinates,
        save_path=os.path.join(out_dir, f"reconstruction_rh_{occ}.png"),
        method_label="RH-NBV",
    ), "reconstruction")

    if recon_snapshots:
        _try(lambda: plot_reconstruction_evolution_grid(
            voxel_snapshots=recon_snapshots,
            mesh_coordinates=rh.mesh_coordinates,
            save_path=os.path.join(out_dir, f"reconstruction_evolution_rh_{occ}.png"),
            method_label="RH-NBV",
        ), "reconstruction_evolution")

        iter_dir = os.path.join(out_dir, "reconstruction_per_iter")
        os.makedirs(iter_dir, exist_ok=True)
        for i, snap in enumerate(recon_snapshots):
            _try(lambda snap=snap, i=i: plot_reconstruction_single_iter(
                target_voxels=snap,
                mesh_coordinates=rh.mesh_coordinates,
                iteration=i + 1,
                save_path=os.path.join(iter_dir, f"reconstruction_rh_{occ}_view{i+1:02d}.png"),
                method_label="RH-NBV",
            ), f"reconstruction_view{i+1}")


def run_single_trial(trial_idx, occ, run_dir):
    trial_seed = BASE_SEED + trial_idx
    trial_dir = os.path.join(run_dir, f"trial_{trial_idx:02d}")
    os.makedirs(trial_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TRIAL {trial_idx + 1}/{NUM_TRIALS}  |  seed={trial_seed}")
    print(f"  K={K}, H={H}, gamma={RH_PARAMS['discount']}, "
          f"lambda={RH_PARAMS['lambda_cost']} | Occlusion: {occ}")
    print(f"{'='*60}\n")

    seed_for_start = None if trial_idx == 0 else trial_seed
    vp = ViewpointPlanning(lr=0, trial_seed=seed_for_start, rh_params=RH_PARAMS)

    recon_snapshots = []
    for i in range(NUM_ITERS):
        print(f"--- RH Iteration {i + 1}/{NUM_ITERS} ---")
        vp._diagnose_f1 = (EXPERIMENT == "D" and i == NUM_ITERS - 1)
        vp.run_rh()
        rh = vp.rh_planner
        snap = rh.target_voxels
        snap = (snap.copy() if isinstance(snap, np.ndarray) and snap.ndim == 2
                else np.zeros((0, 3)))
        recon_snapshots.append(snap)

    target_voxels = vp.rh_planner.target_voxels
    mesh_coords = vp.rh_planner.mesh_coordinates
    if isinstance(target_voxels, np.ndarray) and target_voxels.ndim < 2:
        target_voxels = None

    results = compute_all_metrics(
        coverages=vp.coverages_rh.tolist(),
        recalls=vp.recall_rh.tolist(),
        precisions=vp.precision_rh.tolist(),
        distances=vp.trajectory_distance_rh.tolist(),
        times=vp.cumulative_time_rh.tolist(),
        ray_calls=vp.ray_calls_rh.tolist(),
        method_name="RH-NBV",
        occlusion_type=occ,
        params={**RH_PARAMS, "trial": trial_idx, "seed": trial_seed},
        target_voxels=target_voxels,
        mesh_coordinates=mesh_coords,
        voxels_seen=vp.voxels_seen_rh.tolist(),
        voxels_total=vp.voxels_total_rh.tolist(),
    )
    results["sigma_series"] = vp.sigma_rh.tolist()
    results["occluded_recall_series"] = vp.occluded_recall_rh.tolist()
    results["tp_series"] = vp.tp_rh.tolist()
    results["fp_series"] = vp.fp_rh.tolist()
    results["fn_series"] = vp.fn_rh.tolist()

    save_and_print(results, prefix=os.path.join(trial_dir, "metrics"),
                   experiment=EXPERIMENT)
    save_plots(vp, occ, trial_dir, recon_snapshots=recon_snapshots)
    return results


def summarize(all_results, occ, run_dir):
    def finals(key):
        return [r[key] for r in all_results]

    summary = {
        "occlusion": occ,
        "num_trials": len(all_results),
        "num_iters": NUM_ITERS,
        "rh_params": RH_PARAMS,
        "per_trial": [
            {
                "trial": i,
                "final_coverage": r["final_coverage"],
                "final_f1": r["final_f1"],
                "final_distance": r["final_distance"],
                "total_ray_calls": r["total_ray_calls"],
                "coverage_auc": r["coverage_auc"],
                "tp_final": r["tp_series"][-1],
                "fp_final": r["fp_series"][-1],
                "fn_final": r["fn_series"][-1],
            }
            for i, r in enumerate(all_results)
        ],
        "mean": {},
        "std": {},
    }
    for key in ["final_coverage", "final_f1", "final_distance",
                "total_ray_calls", "coverage_auc"]:
        vals = np.array(finals(key), dtype=float)
        summary["mean"][key] = round(float(vals.mean()), 4)
        summary["std"][key] = round(float(vals.std()), 4)

    path = os.path.join(run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"  SUMMARY over {len(all_results)} trial(s)  |  Occlusion: {occ}")
    print(f"{'#'*60}")
    print(f"  Final coverage : {summary['mean']['final_coverage']:.2f} "
          f"+/- {summary['std']['final_coverage']:.2f} %")
    print(f"  Final F1       : {summary['mean']['final_f1']*100:.2f} "
          f"+/- {summary['std']['final_f1']*100:.2f} %")
    print(f"  Trajectory dist: {summary['mean']['final_distance']:.3f} "
          f"+/- {summary['std']['final_distance']:.3f} m")
    print(f"  Coverage AUC   : {summary['mean']['coverage_auc']:.2f}")
    print(f"  Saved -> {path}\n")
    return summary


if __name__ == "__main__":
    # ROS 2: init node (replaces rospy.init_node)
    ros2_node.init("rh_test")

    occ = detect_occlusion_type()
    run_dir = make_run_dir(occ)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "experiment": EXPERIMENT,
            "num_iters": NUM_ITERS,
            "num_trials": NUM_TRIALS,
            "base_seed": BASE_SEED,
            "rh_params": RH_PARAMS,
            "occlusion": occ,
            "timestamp": datetime.datetime.now().isoformat(),
        }, f, indent=2)

    exp_name = ("Experiment C / Table I (occlusion-handling)" if EXPERIMENT == "C"
                else "Experiment D / Table II (node reconstruction)")
    print(f"\nRun directory: {run_dir}")
    print(f"Burusa-aligned: {exp_name}")
    print(f"  {NUM_ITERS} viewpoints, {NUM_TRIALS} trial(s)\n")

    all_results = []
    for t in range(NUM_TRIALS):
        all_results.append(run_single_trial(t, occ, run_dir))

    summarize(all_results, occ, run_dir)
    print("\nAll trials complete.")
    ros2_node.shutdown()
