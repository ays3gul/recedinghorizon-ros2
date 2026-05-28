"""
metrics.py — Complete evaluation metrics for NBV planner comparison.

BURUSA ET AL. (ICRA 2024) METRICS:
    1. ROI Coverage (%)           — how completely the ROI was observed
    2. F1-score (%)               — harmonic mean of precision and recall
    3. Precision (%)              — fraction of reconstructed points matching ground truth
    4. Recall (%)                 — fraction of ground truth points reconstructed
    5. Ray-tracing Calls (#)      — computational cost (most expensive operation)
    6. Trajectory Distance (m)    — path efficiency (less = better)
    7. Computation Time (s)       — wall-clock time

HIGH PRIORITY (from GenNBV CVPR 2024, VIN-NBV 2025):
    8. Coverage AUC               — area under coverage curve, captures BOTH speed and final value
    9. Hausdorff Distance (m)     — worst-case reconstruction error (max min-distance)

MEDIUM PRIORITY:
    10. Chamfer Distance (m)      — average bidirectional distance between point clouds
    11. Coverage Efficiency (%/m) — coverage gained per meter of movement

LOW PRIORITY (research contribution):
    12. Stagnation Count (#)      — iterations with no coverage change (detects local optima)
    13. First Detection Iter (#)  — when target was first observed
    14. Scanning Efficiency (#)   — viewpoints needed for target coverage threshold
    15. Views-to-Threshold (#)    — iterations to reach F1 > 0.7 or coverage > 50%
"""

import numpy as np
import json
import os


def compute_all_metrics(coverages, recalls, precisions, distances, times,
                        ray_calls, method_name, occlusion_type, params,
                        target_voxels=None, mesh_coordinates=None):
    """
    Compute all evaluation metrics from raw per-iteration arrays.
    
    Args:
        coverages: list[float] — ROI coverage per iteration (starting from 0)
        recalls: list[float] — recall per iteration
        precisions: list[float] — precision per iteration
        distances: list[float] — cumulative trajectory distance per iteration
        times: list[float] — cumulative computation time per iteration
        ray_calls: list[int] — cumulative ray-tracing calls per iteration
        method_name: str — "RH-NBV", "PSO", or "GradientNBV"
        occlusion_type: str — "easy", "hard", "extreme"
        params: dict — planner parameters
        target_voxels: np.array or None — reconstructed target points (for Chamfer/Hausdorff)
        mesh_coordinates: np.array or None — ground truth mesh points (for Chamfer/Hausdorff)
    """
    n = len(coverages)

    # =========================================================
    # BURUSA METRICS
    # =========================================================

    # F1-scores from precision and recall
    f1_scores = []
    for r, p in zip(recalls, precisions):
        if p + r > 0:
            f1_scores.append(round(2 * p * r / (p + r), 4))
        else:
            f1_scores.append(0.0)

    # =========================================================
    # Coverage AUC
    # =========================================================
    # GenNBV (CVPR 2024) uses this as primary metric because
    # final coverage alone doesn't capture HOW FAST you got there.
    # AUC rewards planners that reach high coverage quickly.
    # Normalized by number of iterations so AUC is in [0, 100].
    coverage_auc = round(float(np.trapz(coverages, dx=1) / max(n - 1, 1)), 2)

    # =========================================================
    # Hausdorff Distance
    # =========================================================
    # Worst-case reconstruction error: max of min-distances.
    # Unlike F1 which uses a binary threshold, Hausdorff shows
    # the single worst point alignment.
    hausdorff_dist = None
    chamfer_dist = None
    if target_voxels is not None and mesh_coordinates is not None:
        if len(target_voxels) > 0 and len(mesh_coordinates) > 0:
            from scipy.spatial import KDTree
            mesh_tree = KDTree(mesh_coordinates)
            voxel_tree = KDTree(target_voxels)

            # Forward: for each reconstructed point, distance to nearest GT
            fwd_dists, _ = mesh_tree.query(target_voxels)
            # Backward: for each GT point, distance to nearest reconstructed
            bwd_dists, _ = voxel_tree.query(mesh_coordinates)

            # Hausdorff: max of all min-distances (both directions)
            hausdorff_dist = round(float(max(np.max(fwd_dists), np.max(bwd_dists))), 6)

            # Chamfer: mean bidirectional distance
            chamfer_dist = round(float((np.mean(fwd_dists) + np.mean(bwd_dists)) / 2), 6)

   
   
   
   
    # Coverage per meter of robot movement.
    # Higher = more efficient path planning.
    final_dist = distances[-1] if distances[-1] > 0 else 0.001
    coverage_efficiency = round(coverages[-1] / final_dist, 2)

   
   
   
   

    # Stagnation count: iterations where coverage didn't increase
    # High stagnation = stuck in local optima (GradientNBV problem)
    stagnation_count = 0
    for i in range(1, n):
        if abs(coverages[i] - coverages[i - 1]) < 0.01:
            stagnation_count += 1

    # First detection: first iteration with coverage > 0
    # Late detection = planner struggles to find initial view of target
    first_detection = n
    for i in range(n):
        if coverages[i] > 0.1:
            first_detection = i
            break

    # Scanning efficiency: viewpoints needed to reach coverage thresholds
    # Fewer viewpoints = more efficient planner
    thresholds = [25, 50, 75]
    views_to_threshold = {}
    for thresh in thresholds:
        reached = None
        for i in range(n):
            if coverages[i] >= thresh:
                reached = i
                break
        views_to_threshold[f"views_to_{thresh}pct"] = reached  # None = never reached

    # Views to F1 threshold
    f1_threshold = 0.1  # 10% F1
    views_to_f1 = None
    for i in range(n):
        if f1_scores[i] >= f1_threshold:
            views_to_f1 = i
            break

    # Coverage at key iterations (Burusa Table format)
    key_iters = [0, 3, 6, 9, min(12, n - 1)]
    coverage_at_key = {str(k): round(coverages[min(k, n - 1)], 2) for k in key_iters}

    # =========================================================
    # BUILD RESULTS DICT
    # =========================================================
    results = {
        "method": method_name,
        "occlusion": occlusion_type,
        "params": params,
        "num_iters": n - 1,

        # Per-iteration arrays
        "coverages": [round(c, 2) for c in coverages],
        "f1_scores": f1_scores,
        "recalls": [round(r, 4) for r in recalls],
        "precisions": [round(p, 4) for p in precisions],
        "distances": [round(d, 4) for d in distances],
        "times": [round(t, 1) for t in times],
        "ray_tracing_calls": ray_calls,

        # Burusa summary
        "final_coverage": round(coverages[-1], 2),
        "final_f1": f1_scores[-1],
        "final_distance": round(distances[-1], 4),
        "total_time": round(times[-1], 1),
        "total_ray_calls": ray_calls[-1],
        
        # Visibility efficiency: fraction of max achievable coverage reached.
        # Loads max from none-scenario JSON if available, else uses current run.
        _max_cov = coverages[-1]
        for _p in ["results_rh_nbv_none.json", "results/results_rh_nbv_none.json"]:
            if os.path.exists(_p):
                try:
                    _max_cov = max(json.load(open(_p)).get("final_coverage", _max_cov), 0.01)
                except Exception:
                    pass
                break
        _vis_eff = round(coverages[-1] / _max_cov * 100, 1)


        "visibility_efficiency": _vis_eff,

        # High priority
        "coverage_auc": coverage_auc,
        "hausdorff_distance": hausdorff_dist,
        "chamfer_distance": chamfer_dist,

        # Medium priority
        "coverage_efficiency": coverage_efficiency,

        # Low priority
        "stagnation_count": stagnation_count,
        "first_detection_iter": first_detection,
        "views_to_threshold": views_to_threshold,
        "views_to_f1_threshold": views_to_f1,
        "coverage_at_key_iters": coverage_at_key,
    }

    return results


def detect_occlusion_type(config_path="viewpoint_planners/viewpoint_planning.py"):
    """Detect which occlusion scenario is active from the config file."""
    occlusion_type = "unknown"
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "spawn_extreme_occlusion" in stripped:
                    return "extreme"
                if "spawn_complex_occlusion" in stripped:
                    return "complex3"
                if "spawn_easy_occlusion" in stripped:
                    return "easy"
                if "spawn_hard_occlusion" in stripped:
                    return "hard"
                if "spawn_no_occlusion" in stripped:
                    return "none"
                if "spawn_c_shape_occlusion" in stripped:
                    return "Hard C shaped"
                #if "occlusion_position" in stripped:
                   # if "0.65" in stripped: return "easy"
                   # elif "0.6, -0.25" in stripped: return "hard"
                   # elif "1.25" in stripped: return "top"
                   # elif "0.95" in stripped: return "bottom"
                   # elif "0.35" in stripped: return "right"
    return occlusion_type


def save_and_print(results, prefix="results"):
    """
    Print results in Burusa et al. (ICRA 2024) Table II format.
    Single unified table: rows = metrics, columns = viewpoint indices.
    """
    method  = results["method"]
    occ     = results["occlusion"]
    n       = results["num_iters"]

    covs      = results["coverages"]
    f1s       = [v * 100 for v in results["f1_scores"]]
    ray_calls = results["ray_tracing_calls"]
    dists     = results["distances"]
    sigmas    = results.get("sigma_series", None)

    # Pick columns: 0 + evenly spaced up to n, max 7 columns total
    if n <= 5:
        col_indices = list(range(n + 1))
    else:
        step = n // 5
        col_indices = sorted(set([0] + list(range(step, n, step)) + [n]))
        col_indices = col_indices[:7]

    col_w  = 9
    lbl_w  = 42
    sep    = "=" * (lbl_w + col_w * len(col_indices) + 2)
    dash   = "-" * (lbl_w + col_w * len(col_indices) + 2)

    def _header():
        h = f"  {'# Viewpoints':<{lbl_w}}"
        for i in col_indices:
            h += f"{i:>{col_w}}"
        return h

    def _float_row(label, values, fmt=".1f", bold_last=True):
        row = f"  {label:<{lbl_w}}"
        for k, i in enumerate(col_indices):
            v = values[i] if i < len(values) else None
            if v is None:
                row += f"{'  -':>{col_w}}"
            else:
                cell = f"{v:{col_w}{fmt}}"
                # Bold last column with asterisk marker
                if bold_last and k == len(col_indices) - 1:
                    cell = f"{v:{col_w-1}{fmt}}*"
                row += cell
        return row

    def _int_row(label, values, bold_last=True):
        row = f"  {label:<{lbl_w}}"
        for k, i in enumerate(col_indices):
            v = values[i] if i < len(values) else None
            if v is None:
                row += f"{'  -':>{col_w}}"
            else:
                cell = f"{v:{col_w}d}"
                if bold_last and k == len(col_indices) - 1:
                    cell = f"{v:{col_w-1}d}*"
                row += cell
        return row

    def _dash_row(label):
        return f"  {label:<{lbl_w}}" + f"{'  -':>{col_w}}" * len(col_indices)

    print(f"\n{sep}")
    print(f"  TABLE: {method}  |  Occlusion: {occ}  |  {n} iterations")
    print(f"  (* = final value)")
    print(sep)
    print(_header())
    print(dash)

    # ROI Coverage
    print(_float_row("ROI coverage (%) ↑", covs, fmt=".1f"))
    print(dash)

    # F1
    print(_float_row("F1-score 3D reconstruction (%) ↑", f1s, fmt=".1f"))
    print(dash)

    # Ray-tracing calls
    print(_int_row("Ray-tracing calls (#) ↓", ray_calls))
    print(dash)

    # Trajectory distance
    print(_float_row("Trajectory distance (m) ↓", dists, fmt=".3f"))
    print(dash)

    # Recall occluded
    #print(_dash_row("Recall occluded node (%) ↑"))
    #print(dash)
    
    # Recall occluded
    occ_recalls = results.get("occluded_recall_series", None)
    if occ_recalls is not None:
        occ_pct = [round(v * 100, 1) for v in occ_recalls]
        print(_float_row("Recall occluded node (%) ↑", occ_pct, fmt=".1f"))
    else:
        print(_dash_row("Recall occluded node (%) ↑"))
    print(dash)
 
    # Sigma
    if sigmas is not None:
        sigmas_cm = [round(s * 100, 2) for s in sigmas]
        print(_float_row("sigma 3D node pos (m×10⁻²) ↓", sigmas_cm, fmt=".2f"))
    else:
        print(_dash_row("sigma 3D node pos (m×10⁻²) ↓"))
 
    print(sep)



    # Supporting metrics block
    print(f"\n  SUPPORTING METRICS")
    print(dash)
    print(f"  {'Recall (full mesh, %)':<{lbl_w}} {results['recalls'][-1]*100:>{col_w}.1f}")
    print(f"  {'Precision (full mesh, %)':<{lbl_w}} {results['precisions'][-1]*100:>{col_w}.1f}")
    
    ve = results.get('visibility_efficiency', None)
    if ve is not None:
        print(f"  {'Visibility efficiency (%)':<{lbl_w}} {ve:>{col_w}.1f}")

    print(f"  {'Coverage AUC':<{lbl_w}} {results['coverage_auc']:>{col_w}.1f}")
    print(f"  {'Computation time (s)':<{lbl_w}} {results['total_time']:>{col_w}.1f}")
    if results['hausdorff_distance'] is not None:
        print(f"  {'Hausdorff distance (m)':<{lbl_w}} {results['hausdorff_distance']:>{col_w}.6f}")
        print(f"  {'Chamfer distance (m)':<{lbl_w}} {results['chamfer_distance']:>{col_w}.6f}")
    print(dash)

    # Save JSON
    fname = f"{prefix}_{method.lower().replace('-','_')}_{occ}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {fname}\n")

    return fname
