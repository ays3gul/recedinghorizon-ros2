#!/usr/bin/env python3
"""
test_gradient_node.py — Burusa GradientNBV baseline driver.

Runs Burusa et al.'s gradient-based NBV planner under EXACTLY the same
conditions as test_rh_node.py (same VoxelGrid, same bunny mesh, same F1 metric,
same occlusion scenarios, same output layout) so RH-NBV and GradientNBV are
directly comparable.

This also serves as a diagnostic: if F1 is 0 here too, the issue is in the
shared reconstruction/metric pipeline, not in RH-NBV specifically.

Usage:
    python3 test_gradient_node.py                 # default 5 viewpoints (Exp D)
    EXPERIMENT=C python3 test_gradient_node.py    # 20 viewpoints (Exp C)
"""
import os
import json
import math
import time
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")

import rospy
from scipy.spatial import KDTree

from abb_control.arm_control_client import ArmControlClient
from perception.perceiver import Perceiver
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from utils.py_utils import numpy_to_pose
from utils.sdf_spawner import SDFSpawner
# gradient_nbv_planner.py may sit next to this script (src/) or inside the
# viewpoint_planners/ package — make sure the script's own directory is on the
# import path, then try both locations.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from gradient_nbv_planner import GradientNBVPlanner
except ModuleNotFoundError:
    from viewpoint_planners.gradient_nbv_planner import GradientNBVPlanner

from metrics import compute_all_metrics, detect_occlusion_type, save_and_print

# Same plotting helpers RH-NBV uses, so GradientNBV produces directly
# comparable figures (coverage curve, 3D trajectory, reconstruction vs GT).
# Candidate-sequence plots are RH-specific (GradientNBV has no K candidates),
# so they are intentionally omitted.
from plots.plot_coverage import plot_coverage_progression
from plots.plot_trajectory_3d import plot_3d_trajectory
from plots.plot_reconstruction import plot_reconstruction_comparison

EXPERIMENT = os.environ.get("EXPERIMENT", "D").upper()
_default_iters = 20 if EXPERIMENT == "C" else 5
NUM_ITERS  = int(os.environ.get("NUM_ITERS", _default_iters))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", 1))
BASE_SEED  = int(os.environ.get("BASE_SEED", 42))

# Target node on the bunny body, camera-facing and observable. MUST match
# viewpoint_planning.py exactly for a fair RH-vs-GradientNBV comparison.
# (0.5,-0.4,1.1) yields non-zero coverage; the densest-region point was on the
# occluded side and gave coverage 0. Overridable via TARGET_POS env.
TARGET_POSITION = np.array([0.5, -0.4, 1.1])
_tp_env = os.environ.get("TARGET_POS")
if _tp_env:
    TARGET_POSITION = np.array([float(v) for v in _tp_env.split(",")])


def get_mesh_coordinates():
    """Identical mesh transform to viewpoint_planning.py (fair GT)."""
    file_path = "/home/ayse/gradientnbv/src/simulation_environment/meshes/bunny.dae"
    tree = ET.parse(file_path); root = tree.getroot()
    ns = {"ns": "http://www.collada.org/2005/11/COLLADASchema"}
    arr = root.find(".//ns:float_array[@id='bun_zipper-mesh-positions-array']", ns)
    raw = list(map(float, arr.text.split()))
    vertices = np.array(raw).reshape(-1, 3)
    vertices_swapped = vertices[:, [0, 2, 1]]
    scale = np.array([-1.2, 1.2, 1.2])
    z_corr = float(os.environ.get("MESH_Z_CORR", 0.048))
    translation = np.array([0.5, -0.4, 1.0 - z_corr])
    coords = vertices_swapped * scale + translation
    return coords, KDTree(coords)


# Occlusion scenarios (identical to viewpoint_planning.py)
def spawn_occlusion(sdf, occ):
    if occ == "none":
        pass
    elif occ == "easy":
        sdf.spawn_box(np.array([0.65, -0.3, 1.1]), 1)
    elif occ == "hard":
        sdf.spawn_box(np.array([0.6, -0.25, 1.1]), 1)
    elif occ == "extreme":
        sdf.spawn_box(np.array([0.6, -0.3, 1.1]), 1)
        sdf.spawn_box(np.array([0.6, -0.3, 1.2]), 2)
    elif occ in ("complex", "complex3"):
        sdf.spawn_box(np.array([0.73, -0.25, 0.95]), 1)
        sdf.spawn_bar(np.array([0.5, -0.22, 1.0]), 2)
        sdf.spawn_box(np.array([0.6, -0.32, 1.3]), 3)


def make_run_dir(occ):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}_exp{EXPERIMENT}_{occ}_GradientNBV"
    d = os.path.join("results", name)
    os.makedirs(d, exist_ok=True)
    return d


def run_single_trial(trial_idx, occ, run_dir, mesh_coords, mesh_tree,
                     arm, perceiver, sampler):
    trial_dir = os.path.join(run_dir, f"trial_{trial_idx:02d}")
    os.makedirs(trial_dir, exist_ok=True)
    print(f"\n{'='*60}\n  GradientNBV TRIAL {trial_idx+1}/{NUM_TRIALS} | Occlusion: {occ}\n{'='*60}\n")

    start_pose = sampler.predefine_start_pose(TARGET_POSITION)
    arm.move_arm_to_pose(numpy_to_pose(start_pose))

    cam_info = perceiver.get_camera_info()
    image_size = np.array([cam_info.width, cam_info.height])
    intrinsics = np.array(cam_info.K).reshape(3, 3)

    planner = GradientNBVPlanner(
        start_pose=start_pose,
        grid_size=np.array([0.3, 0.6, 0.3]),
        grid_center=TARGET_POSITION,
        image_size=image_size,
        intrinsics=intrinsics,
        target_params=TARGET_POSITION,
        mesh_coordinates=mesh_coords,
        mesh_tree=mesh_tree,
    )

    coverages = [0.0]; recalls = [0.0]; precisions = [0.0]
    distances = [0.0]; times = [0.0]; ray_calls = [0]
    tp = [0]; fp = [0]; fn = [0]
    trail = [start_pose[:3].copy()]

    for i in range(NUM_ITERS):
        print(f"--- GradientNBV Iteration {i+1}/{NUM_ITERS} ---")
        t0 = time.time()
        viewpoint, loss, _ = planner.next_best_view(target_pos=TARGET_POSITION)
        ok = arm.move_arm_to_pose(numpy_to_pose(viewpoint))
        rospy.sleep(1.0)
        if ok:
            depth, _, sem = perceiver.run()
            cov = planner.update_voxel_grid(depth, sem, viewpoint)
            cov = float(cov) if cov is not None else coverages[-1]
            d = math.sqrt(sum((viewpoint[k]-trail[-1][k])**2 for k in range(3)))
            trail.append(viewpoint[:3].copy())
            distances.append(distances[-1] + d)
        else:
            cov = coverages[-1]
            distances.append(distances[-1])
        coverages.append(cov)
        times.append(times[-1] + (time.time() - t0))
        diag = (EXPERIMENT == "D" and i == NUM_ITERS - 1)
        f1, rec, prec = planner.calculate_F1(diagnose=diag)
        recalls.append(rec); precisions.append(prec)
        ray_calls.append(planner.ray_trace_count)
        tp.append(planner.last_tp); fp.append(planner.last_fp); fn.append(planner.last_fn)
        print(f"[GradNBV] coverage={cov:.4f} | loss={float(loss):.4f} | "
              f"F1={f1:.4f} | recall={rec:.4f} | precision={prec:.4f}")

    results = compute_all_metrics(
        coverages=coverages, recalls=recalls, precisions=precisions,
        distances=distances, times=times, ray_calls=ray_calls,
        method_name="GradientNBV", occlusion_type=occ,
        params={"planner": "GradientNBV", "lr": 0.03, "trial": trial_idx},
        target_voxels=planner.target_voxels, mesh_coordinates=mesh_coords,
    )
    results["tp_series"] = tp; results["fp_series"] = fp; results["fn_series"] = fn
    save_and_print(results, prefix=os.path.join(trial_dir, "metrics"),
                   experiment=EXPERIMENT)

    # --- Plots (same helpers as RH-NBV, labelled GradientNBV) ---
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

    return results


if __name__ == "__main__":
    rospy.init_node("gradient_test")
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
                   "num_iters": NUM_ITERS, "occlusion": occ,
                   "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)

    print(f"\nRun directory: {run_dir}")
    print(f"GradientNBV baseline | {NUM_ITERS} viewpoints | Occlusion: {occ}\n")

    all_results = [run_single_trial(t, occ, run_dir, mesh_coords, mesh_tree,
                                    arm, perceiver, sampler)
                   for t in range(NUM_TRIALS)]
    print("\nGradientNBV baseline complete.")
