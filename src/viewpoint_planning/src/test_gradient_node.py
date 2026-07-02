#!/usr/bin/env python3
"""
test_gradient_node.py — ROS 2 Jazzy / Burusa GradientNBV baseline driver.

Usage:
    OCC=well ./run_ros2_gz.sh SCRIPT=test_gradient_node.py
    OCC=frontal EXPERIMENT=C ./run_ros2_gz.sh SCRIPT=test_gradient_node.py
"""
import os
import json
import math
import time
import datetime
import xml.etree.ElementTree as ET

# Absolute results root so ROS2's CWD changes don't break relative paths
_RESULTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
)

import numpy as np
import matplotlib
matplotlib.use("Agg")

from scipy.spatial import KDTree

import ros2_node  # ROS 2 singleton (replaces rospy.init_node)

from abb_control.arm_control_client import ArmControlClient
from perception.perceiver import Perceiver
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from utils.py_utils import numpy_to_pose
from utils.sdf_spawner import SDFSpawner

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from gradient_nbv_planner import GradientNBVPlanner
except ModuleNotFoundError:
    from viewpoint_planners.gradient_nbv_planner import GradientNBVPlanner

from metrics import compute_all_metrics, detect_occlusion_type, save_and_print

try:
    from fair_comparison_config import (
        GRID_SIZE as FC_GRID_SIZE,
        get_target_position as fc_get_target_position,
        jitter_start_pose as fc_jitter_start_pose,
        seed_for_trial as fc_seed_for_trial,
    )
except ModuleNotFoundError:
    from viewpoint_planners.fair_comparison_config import (
        GRID_SIZE as FC_GRID_SIZE,
        get_target_position as fc_get_target_position,
        jitter_start_pose as fc_jitter_start_pose,
        seed_for_trial as fc_seed_for_trial,
    )

from plots.plot_coverage import plot_coverage_progression
from plots.plot_trajectory_3d import plot_3d_trajectory
from plots.plot_reconstruction import (
    plot_reconstruction_comparison,
    plot_reconstruction_evolution_grid,
    plot_reconstruction_single_iter,
)

EXPERIMENT = os.environ.get("EXPERIMENT", "D").upper()
_default_iters = 20 if EXPERIMENT == "C" else 5
NUM_ITERS  = int(os.environ.get("NUM_ITERS", _default_iters))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", 1))
BASE_SEED  = int(os.environ.get("BASE_SEED", 42))

TARGET_POSITION = fc_get_target_position()


def get_mesh_coordinates():
    """Identical mesh transform to viewpoint_planning.py (fair GT)."""
    meshes = "/home/ayse/Desktop/RecedingHorizon/src/simulation_environment/meshes"
    ns = {"ns": "http://www.collada.org/2005/11/COLLADASchema"}
    target = os.environ.get("TARGET", "bunny").lower()

    if target == "mug":
        root = ET.parse(f"{meshes}/coffee_mug.dae").getroot()
        arr = root.find(".//ns:float_array[@id='coffee_mug-mesh-positions-array']", ns)
        vertices = np.array(list(map(float, arr.text.split()))).reshape(-1, 3)
        coords = vertices + np.array([0.5, -0.30, 1.0])
    elif target == "tomato":
        root = ET.parse(f"{meshes}/tomato6.dae").getroot()
        fruit_nodes = {"Fruit1", "Fruit2", "Fruit3", "Fruit4"}
        fruit_arr_ids = set()
        for node in root.findall(".//ns:visual_scene//ns:node", ns):
            if node.get("name", "") in fruit_nodes:
                for inst in node.findall(".//ns:instance_geometry", ns):
                    url = inst.get("url", "").lstrip("#")
                    fruit_arr_ids.add(url.replace("-mesh", "") + "-mesh-positions-array")
        all_verts = []
        for fa in root.findall(".//ns:float_array", ns):
            if fa.get("id", "") in fruit_arr_ids:
                verts = np.array(list(map(float, fa.text.split()))).reshape(-1, 3)
                all_verts.append(verts)
        vertices = np.vstack(all_verts)
        # COLLADA Y-up → Gazebo Z-up: world=(dae_x, -dae_z, dae_y)
        vertices_converted = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
        coords = vertices_converted * 0.4 + np.array([0.5, -0.50, 0.9])
    else:
        root = ET.parse(f"{meshes}/bunny.dae").getroot()
        arr = root.find(".//ns:float_array[@id='bun_zipper-mesh-positions-array']", ns)
        vertices = np.array(list(map(float, arr.text.split()))).reshape(-1, 3)
        vertices_converted = np.column_stack([
            -vertices[:, 0],
            vertices[:, 2],
            vertices[:, 1] - 0.05,
        ])
        scale = np.array([1.2, 1.2, 1.2])
        coords = vertices_converted * scale + np.array([0.5, -0.30, 1.0])

    return coords, KDTree(coords)


def spawn_occlusion(sdf, occ):
    pass  # panels defined in world SDF files (ur5e_world_<occ>.sdf)


def make_run_dir(occ):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}_exp{EXPERIMENT}_{occ}_GradientNBV"
    d = os.path.join(_RESULTS_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def run_single_trial(trial_idx, occ, run_dir, mesh_coords, mesh_tree,
                     arm, perceiver, sampler):
    trial_dir = os.path.join(run_dir, f"trial_{trial_idx:02d}")
    os.makedirs(trial_dir, exist_ok=True)
    print(f"\n{'='*60}\n  GradientNBV TRIAL {trial_idx+1}/{NUM_TRIALS} | Occlusion: {occ}\n{'='*60}\n")

    start_pose = sampler.predefine_start_pose(TARGET_POSITION)
    start_pose = fc_jitter_start_pose(start_pose, trial_idx)
    arm.move_arm_to_pose(numpy_to_pose(start_pose))

    cam_info = perceiver.get_camera_info()
    image_size = np.array([cam_info.width, cam_info.height])
    intrinsics = np.array(cam_info.k).reshape(3, 3)  # ROS 2: lowercase k

    planner = GradientNBVPlanner(
        start_pose=start_pose,
        grid_size=np.array(FC_GRID_SIZE, dtype=float),
        grid_center=TARGET_POSITION,
        image_size=image_size,
        intrinsics=intrinsics,
        target_params=TARGET_POSITION,
        mesh_coordinates=mesh_coords,
        mesh_tree=mesh_tree,
    )

    # Set occluded baseline on the empty grid (before any planning) so
    # that every planner starts from an identical 100% occluded baseline.
    planner.set_occluded_mesh_points()

    coverages = [0.0]; sem_coverages = [0.0]; recalls = [0.0]; precisions = [0.0]
    distances = [0.0]; times = [0.0]; ray_calls = [0]
    tp = [0]; fp = [0]; fn = [0]
    sigmas = [0.0]; occ_recalls = [0.0]
    voxels_seen = [0]; voxels_total = [0]
    recon_snapshots = []
    trail = [start_pose[:3].copy()]

    for i in range(NUM_ITERS):
        print(f"--- GradientNBV Iteration {i+1}/{NUM_ITERS} ---")
        t0 = time.time()
        viewpoint, loss, _ = planner.next_best_view(target_pos=TARGET_POSITION)
        ok = arm.move_arm_to_pose(numpy_to_pose(viewpoint))
        time.sleep(1.0)  # ROS 2: time.sleep instead of rospy.sleep
        if ok:
            # Use commanded viewpoint for integration. The TF actual pose
            # (camera_color_frame) uses z=forward in Gazebo, but T_oc expects
            # x=forward — mismatched convention causes ~43% FP. Commanded pose
            # from look_at_rotation (ref=[1,0,0]) is internally consistent with
            # T_oc and gives precision ~0.99 (verified June 9 runs).
            integ_vp = viewpoint
            depth, _, sem = perceiver.run()
            if depth is not None and sem is not None:
                cov = planner.update_voxel_grid(depth, sem, integ_vp)
                cov = float(cov) if cov is not None else coverages[-1]
            else:
                cov = coverages[-1]
            sem_cov = planner.voxel_grid.semantic_coverage
            voxels_seen.append(planner.voxel_grid.n_seen)
            voxels_total.append(planner.voxel_grid.n_total)
            d = math.sqrt(sum((viewpoint[k]-trail[-1][k])**2 for k in range(3)))
            trail.append(viewpoint[:3].copy())
            distances.append(distances[-1] + d)
        else:
            cov = coverages[-1]
            sem_cov = sem_coverages[-1]
            voxels_seen.append(voxels_seen[-1])
            voxels_total.append(voxels_total[-1])
            distances.append(distances[-1])
        coverages.append(cov)
        sem_coverages.append(sem_cov)
        times.append(times[-1] + (time.time() - t0))

        diag = (EXPERIMENT == "D" and i == NUM_ITERS - 1)
        occ_positions = None
        if occ == "tunnel":
            occ_positions = [
                (np.array([0.425, -0.23, 1.07]), np.array([0.035, 0.012, 0.150])),
                (np.array([0.575, -0.23, 1.07]), np.array([0.035, 0.012, 0.150])),
            ]
        elif occ == "partial":
            # panel_partial: center (0.45, -0.219, 1.07), size 0.10x0.02x0.20
            occ_positions = [
                (np.array([0.45, -0.219, 1.07]), np.array([0.05, 0.01, 0.10])),
            ]
        elif occ == "well":
            # side_1 (0.40,-0.30,1.08) 0.02x0.22x0.16
            # side_2 (0.64,-0.30,1.08) 0.02x0.22x0.16
            # front  (0.52,-0.18,1.08) 0.26x0.02x0.16
            # back   (0.52,-0.42,1.08) 0.26x0.02x0.16
            occ_positions = [
                (np.array([0.40, -0.30, 1.08]), np.array([0.010, 0.110, 0.080])),
                (np.array([0.64, -0.30, 1.08]), np.array([0.010, 0.110, 0.080])),
                (np.array([0.52, -0.18, 1.08]), np.array([0.130, 0.010, 0.080])),
                (np.array([0.52, -0.42, 1.08]), np.array([0.130, 0.010, 0.080])),
            ]
        f1, rec, prec = planner.calculate_F1(
            occluder_positions=occ_positions, diagnose=diag)
        recalls.append(rec); precisions.append(prec)
        ray_calls.append(planner.ray_trace_count)
        tp.append(planner.last_tp); fp.append(planner.last_fp); fn.append(planner.last_fn)
        sigmas.append(planner.compute_sigma())
        occ_recalls.append(planner.compute_occluded_recall())

        snap = planner.target_voxels
        recon_snapshots.append(
            snap.copy() if isinstance(snap, np.ndarray) and snap.ndim == 2
            else np.zeros((0, 3)))

        print(f"[GradNBV] coverage={cov:.4f} | sem_coverage={sem_cov:.4f} | "
              f"loss={float(loss):.4f} | F1={f1:.4f} | recall={rec:.4f} | "
              f"precision={prec:.4f} | occ_recall={occ_recalls[-1]:.4f}")
        planner.visualize()

    # DEBUG: dump reconstruction occupied points (world frame) for mesh-alignment.
    try:
        _occ = planner.voxel_grid.get_occupied_points()[0].detach().cpu().numpy()
        np.save(os.path.join(trial_dir, "recon_points.npy"), _occ)
        np.save(os.path.join(trial_dir, "mesh_coords.npy"), mesh_coords)
        print(f"[DEBUG] saved recon_points {_occ.shape}, mesh_coords {mesh_coords.shape}")
    except Exception as _e:
        print("[DEBUG] recon dump failed:", _e)

    results = compute_all_metrics(
        coverages=coverages, recalls=recalls, precisions=precisions,
        distances=distances, times=times, ray_calls=ray_calls,
        method_name="GradientNBV", occlusion_type=occ,
        params={"planner": "GradientNBV", "lr": 0.03, "trial": trial_idx,
                "seed": fc_seed_for_trial(trial_idx)},
        target_voxels=planner.target_voxels, mesh_coordinates=mesh_coords,
        voxels_seen=voxels_seen, voxels_total=voxels_total,
    )
    results["tp_series"] = tp; results["fp_series"] = fp; results["fn_series"] = fn
    results["sigma_series"] = sigmas
    results["occluded_recall_series"] = occ_recalls
    results["semantic_coverage_series"] = sem_coverages
    save_and_print(results, prefix=os.path.join(trial_dir, "metrics"),
                   experiment=EXPERIMENT)

    def _try(fn, label):
        try:
            fn()
        except Exception as e:
            print(f"  [plot] {label} failed: {e}")

    _try(lambda: plot_coverage_progression(
        coverages={"GradientNBV": coverages},
        save_path=os.path.join(trial_dir, f"coverage_gradientnbv_{occ}.png"),
        title=f"GradientNBV Coverage (Occlusion: {occ})",
    ), "coverage")

    _try(lambda: plot_3d_trajectory(
        trail=trail,
        mesh_coordinates=mesh_coords,
        occlusion_type=occ,
        save_path=os.path.join(trial_dir, f"trajectory_3d_gradientnbv_{occ}.png"),
        title=f"GradientNBV 3D Trajectory (Occlusion: {occ})",
        method_label="GradientNBV",
    ), "trajectory")

    _try(lambda: plot_reconstruction_comparison(
        target_voxels=planner.target_voxels,
        mesh_coordinates=mesh_coords,
        save_path=os.path.join(trial_dir, f"reconstruction_gradientnbv_{occ}.png"),
        method_label="GradientNBV",
    ), "reconstruction")

    if recon_snapshots:
        _try(lambda: plot_reconstruction_evolution_grid(
            voxel_snapshots=recon_snapshots,
            mesh_coordinates=mesh_coords,
            save_path=os.path.join(trial_dir,
                                   f"reconstruction_evolution_gradientnbv_{occ}.png"),
            method_label="GradientNBV",
        ), "reconstruction_evolution")

        iter_dir = os.path.join(trial_dir, "reconstruction_per_iter")
        os.makedirs(iter_dir, exist_ok=True)
        for i, snap in enumerate(recon_snapshots):
            _try(lambda snap=snap, i=i: plot_reconstruction_single_iter(
                target_voxels=snap,
                mesh_coordinates=mesh_coords,
                iteration=i + 1,
                save_path=os.path.join(iter_dir,
                                       f"reconstruction_gradientnbv_{occ}_view{i+1:02d}.png"),
                method_label="GradientNBV",
            ), f"reconstruction_view{i+1}")

    return results


def summarize(all_results, occ, run_dir):
    def finals(key):
        return [r[key] for r in all_results]

    summary = {
        "occlusion": occ,
        "num_trials": len(all_results),
        "num_iters": NUM_ITERS,
        "planner": "GradientNBV",
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
    ros2_node.init("gradient_test")

    arm = ArmControlClient()
    perceiver = Perceiver()
    sampler = ViewpointSampler()
    sdf = SDFSpawner()

    occ = detect_occlusion_type()
    spawn_occlusion(sdf, occ)

    mesh_coords, mesh_tree = get_mesh_coordinates()
    run_dir = make_run_dir(occ)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"experiment": EXPERIMENT, "planner": "GradientNBV",
                   "num_iters": NUM_ITERS, "num_trials": NUM_TRIALS,
                   "base_seed": BASE_SEED,
                   "grid_size": [float(v) for v in FC_GRID_SIZE],
                   "occlusion": occ,
                   "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)

    print(f"\nRun directory: {run_dir}")
    print(f"GradientNBV baseline | {NUM_ITERS} viewpoints | Occlusion: {occ}\n")

    all_results = [run_single_trial(t, occ, run_dir, mesh_coords, mesh_tree,
                                    arm, perceiver, sampler)
                   for t in range(NUM_TRIALS)]
    summarize(all_results, occ, run_dir)
    print("\nGradientNBV baseline complete.")
    ros2_node.shutdown()
