#!/usr/bin/env python3

import contextlib
import csv
import importlib
import importlib.util
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import rospy

from viewpoint_planners.viewpoint_planning import ViewpointPlanning
from metrics import compute_all_metrics, detect_occlusion_type, save_and_print


NUM_ITERS = 12
SILENCE_INTERNAL_LOGS = True
PLOT_DIR = os.path.join("results", "plots")


def _load_plot_function(function_name, module_candidates, file_candidates):
    """Load a plotting function from normal modules or direct .py paths."""
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, function_name, None)
            if fn is not None:
                return fn
        except Exception:
            pass

    for file_path in file_candidates:
        if not file_path or not os.path.exists(file_path):
            continue
        try:
            safe_name = "_dynamic_" + os.path.basename(file_path).replace(".", "_").replace("(", "_").replace(")", "_")
            spec = importlib.util.spec_from_file_location(safe_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, function_name, None)
            if fn is not None:
                return fn
        except Exception as exc:
            print(f"[PlotLoader] Could not load {function_name} from {file_path}: {exc}")
    return None


def _find_plot_function(function_name):
    """Find user's existing plot functions regardless of where the files live."""
    cwd = os.getcwd()
    candidates = {
        "plot_coverage_progression": (
            ["plots.plot_coverage", "plot_coverage"],
            [
                os.path.join(cwd, "plots", "plot_coverage.py"),
                os.path.join(cwd, "plot_coverage.py"),
                os.path.join(cwd, "plot_coverage(2).py"),
            ],
        ),
        "plot_3d_trajectory": (
            ["plots.plot_trajectory_3d", "plot_trajectory_3d"],
            [
                os.path.join(cwd, "plots", "plot_trajectory_3d.py"),
                os.path.join(cwd, "plot_trajectory_3d.py"),
                os.path.join(cwd, "plot_trajectory_3d(2).py"),
            ],
        ),
        "plot_reconstruction_comparison": (
            ["plots.plot_reconstruction", "plot_reconstruction"],
            [
                os.path.join(cwd, "plots", "plot_reconstruction.py"),
                os.path.join(cwd, "plot_reconstruction.py"),
                os.path.join(cwd, "plot_reconstruction(2).py"),
            ],
        ),
        "plot_candidate_sequences": (
            ["plots.plot_candidate_sequences", "plot_candidate_sequences"],
            [
                os.path.join(cwd, "plots", "plot_candidate_sequences.py"),
                os.path.join(cwd, "plot_candidate_sequences.py"),
                os.path.join(cwd, "plot_candidate_sequences(1).py"),
            ],
        ),
        "plot_candidate_sequences_grid": (
            ["plots.plot_candidate_sequences", "plot_candidate_sequences"],
            [
                os.path.join(cwd, "plots", "plot_candidate_sequences.py"),
                os.path.join(cwd, "plot_candidate_sequences.py"),
                os.path.join(cwd, "plot_candidate_sequences(1).py"),
            ],
        ),
    }
    modules, files = candidates[function_name]
    return _load_plot_function(function_name, modules, files)


@contextlib.contextmanager
def maybe_silence():
    """Suppress verbose planner prints during one iteration."""
    if not SILENCE_INTERNAL_LOGS:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def line(char="-", n=104):
    print(char * n)


def print_title(occ):
    print("\n" + "=" * 104)
    print(f"COMPARISON: RH-NBV vs GradientNBV | Occlusion: {occ}")
    print("=" * 104)
    print("Higher Coverage/F1/Recall/Precision is better; lower Distance/Time/Ray calls is better.")


def print_method_header(method):
    print(f"\n{method}")
    line()
    print(
        f"{'it':>2} | {'cov%':>8} | {'F1':>7} | {'recall':>7} | {'prec':>7} | "
        f"{'dist(m)':>8} | {'time(s)':>8} | {'rays':>6} | {'eval':>5} | {'move':>4}"
    )
    line()


def latest_summary(vp, method, iteration):
    """Read latest metrics from ViewpointPlanning.

    Compatible with both versions:
    - new ViewpointPlanning: has last_rh_summary / last_gradient_summary
    - old ViewpointPlanning: only has metric arrays
    """
    if method == "RH-NBV":
        last = getattr(vp, "last_rh_summary", {}) or {}
        return {
            "method": method,
            "iteration": iteration,
            "coverage": float(vp.coverages_rh[-1]),
            "f1": float(vp.f1_rh[-1]),
            "recall": float(vp.recall_rh[-1]),
            "precision": float(vp.precision_rh[-1]),
            "distance": float(vp.trajectory_distance_rh[-1]),
            "time": float(vp.cumulative_time_rh[-1]),
            "ray_calls": int(vp.ray_calls_rh[-1]),
            "evals": int(last.get("evals", 0)),
            "success": last.get("success", None),
        }

    last = getattr(vp, "last_gradient_summary", {}) or {}
    return {
        "method": method,
        "iteration": iteration,
        "coverage": float(vp.coverages_grad[-1]),
        "f1": float(vp.f1_grad[-1]),
        "recall": float(vp.recall_grad[-1]),
        "precision": float(vp.precision_grad[-1]),
        "distance": float(vp.trajectory_distance_grad[-1]),
        "time": float(vp.cumulative_time_grad[-1]),
        "ray_calls": int(vp.ray_calls_grad[-1]),
        "evals": int(last.get("evals", 0)),
        "success": last.get("success", None),
    }

def print_row(s):
    if s["success"] is None:
        move = "?"
    else:
        move = "OK" if s["success"] else "FAIL"
    print(
        f"{s['iteration']:>2} | "
        f"{s['coverage']:>8.3f} | "
        f"{s['f1']:>7.4f} | "
        f"{s['recall']:>7.4f} | "
        f"{s['precision']:>7.4f} | "
        f"{s['distance']:>8.3f} | "
        f"{s['time']:>8.1f} | "
        f"{s['ray_calls']:>6} | "
        f"{s['evals']:>5} | "
        f"{move:>4}"
    )


def _latest_ray_calls(vp, method):
    if method == "RH-NBV":
        return int(vp.ray_calls_rh[-1]) if hasattr(vp, "ray_calls_rh") else 0
    return int(vp.ray_calls_grad[-1]) if hasattr(vp, "ray_calls_grad") else 0


def _fill_missing_run_info(summary, ret, prev_rays):
    """Older ViewpointPlanning returns eval count but not last_*_summary."""
    if summary.get("evals", 0) == 0:
        if isinstance(ret, tuple) and len(ret) >= 6:
            try:
                summary["evals"] = int(ret[-1])
            except Exception:
                pass
        if summary.get("evals", 0) == 0:
            summary["evals"] = max(0, int(summary["ray_calls"] - prev_rays))
    return summary

def collect_results(vp, method, occ):
    if method == "RH-NBV":
        coverages = vp.coverages_rh.tolist()
        recalls = vp.recall_rh.tolist()
        precisions = vp.precision_rh.tolist()
        distances = vp.trajectory_distance_rh.tolist()
        times = vp.cumulative_time_rh.tolist()
        ray_calls = vp.ray_calls_rh.tolist()
        target_v = vp.rh_planner.target_voxels
        sigma = vp.sigma_rh.tolist()
        voxels_seen = vp.voxels_seen_rh.tolist() if hasattr(vp, "voxels_seen_rh") else None
        voxels_total = vp.voxels_total_rh.tolist() if hasattr(vp, "voxels_total_rh") else None
        params = {
            "K": 10,
            "H": "adaptive: 2/3/5",
            "step_size": 0.065,
            "lambda": 2.0,
            "r_min": 0.15,
            "r_max": 0.45,
        }
    else:
        coverages = vp.coverages_grad.tolist()
        recalls = vp.recall_grad.tolist()
        precisions = vp.precision_grad.tolist()
        distances = vp.trajectory_distance_grad.tolist()
        times = vp.cumulative_time_grad.tolist()
        ray_calls = vp.ray_calls_grad.tolist()
        target_v = getattr(vp, "target_voxels_grad", None)
        sigma = vp.sigma_grad.tolist()
        voxels_seen = vp.voxels_seen_grad.tolist() if hasattr(vp, "voxels_seen_grad") else None
        voxels_total = vp.voxels_total_grad.tolist() if hasattr(vp, "voxels_total_grad") else None
        params = {"num_samples": 1, "lr": 0.03}

    if isinstance(target_v, np.ndarray) and target_v.ndim < 2:
        target_v = None

    results = compute_all_metrics(
        coverages=coverages,
        recalls=recalls,
        precisions=precisions,
        distances=distances,
        times=times,
        ray_calls=ray_calls,
        method_name=method,
        occlusion_type=occ,
        params=params,
        target_voxels=target_v,
        mesh_coordinates=vp.mesh_coordinates,
        voxels_seen=voxels_seen,
        voxels_total=voxels_total,
    )
    results["sigma_series"] = sigma
    return results


def run_method(method):
    print_method_header(method)
    with maybe_silence():
        vp = ViewpointPlanning()

    rows = []
    for i in range(1, NUM_ITERS + 1):
        prev_rays = _latest_ray_calls(vp, method)
        with maybe_silence():
            if method == "RH-NBV":
                ret = vp.run_rh()
            else:
                ret = vp.run_gradient()
        s = latest_summary(vp, method, i)
        s = _fill_missing_run_info(s, ret, prev_rays)
        rows.append(s)
        print_row(s)
    line()
    return vp, rows

def print_final_comparison(rh_rows, grad_rows):
    rh = rh_rows[-1]
    gr = grad_rows[-1]
    print("\nFINAL SUMMARY")
    line()
    print(f"{'metric':<16} | {'RH-NBV':>12} | {'GradientNBV':>12} | {'winner':>12}")
    line()

    metrics = [
        ("coverage", "higher"),
        ("f1", "higher"),
        ("recall", "higher"),
        ("precision", "higher"),
        ("distance", "lower"),
        ("time", "lower"),
        ("ray_calls", "lower"),
    ]
    for key, direction in metrics:
        a, b = rh[key], gr[key]
        if abs(a - b) < 1e-9:
            winner = "tie"
        elif direction == "higher":
            winner = "RH-NBV" if a > b else "GradientNBV"
        else:
            winner = "RH-NBV" if a < b else "GradientNBV"
        if key == "ray_calls":
            print(f"{key:<16} | {a:>12.0f} | {b:>12.0f} | {winner:>12}")
        else:
            print(f"{key:<16} | {a:>12.4f} | {b:>12.4f} | {winner:>12}")
    line()


def save_live_csv(rows):
    os.makedirs("results", exist_ok=True)
    path = "results/comparison_live_summary.csv"
    fieldnames = [
        "method", "iteration", "coverage", "f1", "recall", "precision",
        "distance", "time", "ray_calls", "evals", "success",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved compact per-iteration table: {path}")


def save_simple_metric_plot(rh_rows, grad_rows, key, ylabel, filename, higher_is_better=True):
    os.makedirs(PLOT_DIR, exist_ok=True)
    x_rh = [r["iteration"] for r in rh_rows]
    y_rh = [r[key] for r in rh_rows]
    x_gr = [r["iteration"] for r in grad_rows]
    y_gr = [r[key] for r in grad_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x_rh, y_rh, marker="o", label="RH-NBV")
    plt.plot(x_gr, y_gr, marker="s", label="GradientNBV")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    suffix = "higher is better" if higher_is_better else "lower is better"
    plt.title(f"{ylabel} per iteration ({suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    return path


def save_existing_style_plots(rh_vp, grad_vp, rh_rows, grad_rows, occ):
    """Reuse the user's existing plotting scripts when available."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    saved = []

    plot_coverage_progression = _find_plot_function("plot_coverage_progression")
    plot_3d_trajectory = _find_plot_function("plot_3d_trajectory")
    plot_reconstruction_comparison = _find_plot_function("plot_reconstruction_comparison")
    plot_candidate_sequences = _find_plot_function("plot_candidate_sequences")
    plot_candidate_sequences_grid = _find_plot_function("plot_candidate_sequences_grid")


    if plot_coverage_progression is not None:
        path = os.path.join(PLOT_DIR, "coverage_existing_style.png")
        fig_ax = plot_coverage_progression(
            {
                "RH-NBV": rh_vp.coverages_rh,
                "Gradient": grad_vp.coverages_grad,
            },
            save_path=path,
            title=f"Coverage Progression — RH-NBV vs GradientNBV ({occ})",
        )
        saved.append(path)
        plt.close("all")
    else:
        saved.append(save_simple_metric_plot(rh_rows, grad_rows, "coverage", "Coverage (%)", "coverage_comparison.png", True))

    # Useful comparison plots, simple style.
    simple_specs = [
        ("f1", "F1 score", "f1_comparison.png", True),
        ("recall", "Recall", "recall_comparison.png", True),
        ("precision", "Precision", "precision_comparison.png", True),
        ("distance", "Trajectory distance (m)", "distance_comparison.png", False),
        ("time", "Cumulative time (s)", "time_comparison.png", False),
        ("ray_calls", "Cumulative ray/gain calls", "ray_calls_comparison.png", False),
    ]
    for key, ylabel, filename, higher in simple_specs:
        saved.append(save_simple_metric_plot(rh_rows, grad_rows, key, ylabel, filename, higher))


    if plot_3d_trajectory is not None:
        for label, trail in [
            ("RH-NBV", np.asarray(rh_vp.trail_rh)),
            ("GradientNBV", np.asarray(grad_vp.trail_grad)),
        ]:
            path = os.path.join(PLOT_DIR, f"trajectory_3d_{label.lower().replace('-', '_')}.png")
            plot_3d_trajectory(
                trail=trail,
                mesh_coordinates=rh_vp.mesh_coordinates,
                occlusion_type=occ,
                save_path=path,
                title=f"{label} 3D Trajectory (Occlusion: {occ})",
                method_label=label,
            )
            saved.append(path)
            plt.close("all")


    if plot_reconstruction_comparison is not None:
        recon_items = [
            ("RH-NBV", getattr(rh_vp.rh_planner, "target_voxels", None)),
            ("GradientNBV", getattr(grad_vp, "target_voxels_grad", None)),
        ]
        for label, voxels in recon_items:
            path = os.path.join(PLOT_DIR, f"reconstruction_{label.lower().replace('-', '_')}.png")
            out = plot_reconstruction_comparison(
                target_voxels=voxels,
                mesh_coordinates=rh_vp.mesh_coordinates,
                save_path=path,
                method_label=label,
            )
            if out != (None, None) and os.path.exists(path):
                saved.append(path)
            plt.close("all")



    candidate_history = getattr(rh_vp.rh_planner, "candidate_history", [])
    if candidate_history and plot_candidate_sequences is not None:
        path = os.path.join(PLOT_DIR, "rh_candidate_sequences_single.png")
        plot_candidate_sequences(
            candidate_history=candidate_history,
            mesh_coordinates=rh_vp.mesh_coordinates,
            occlusion_type=occ,
            iteration_to_plot=min(2, len(candidate_history) - 1),
            save_path=path,
            title=f"RH-NBV Candidate Sequences ({occ})",
        )
        saved.append(path)
        plt.close("all")

    if candidate_history and plot_candidate_sequences_grid is not None:
        path = os.path.join(PLOT_DIR, "rh_candidate_sequences_grid.png")
        plot_candidate_sequences_grid(
            candidate_history=candidate_history,
            mesh_coordinates=rh_vp.mesh_coordinates,
            occlusion_type=occ,
            save_path=path,
        )
        saved.append(path)
        plt.close("all")

    # compact normalized final summary.
    saved.append(save_final_normalized_summary(rh_rows, grad_rows))

    print("\nSaved plots:")
    for path in saved:
        print(f"  - {path}")
    return saved


def save_final_normalized_summary(rh_rows, grad_rows):
    os.makedirs(PLOT_DIR, exist_ok=True)
    final_rh = rh_rows[-1]
    final_gr = grad_rows[-1]
    metrics = ["coverage", "f1", "recall", "precision", "distance", "time", "ray_calls"]
    directions = {
        "coverage": "higher",
        "f1": "higher",
        "recall": "higher",
        "precision": "higher",
        "distance": "lower",
        "time": "lower",
        "ray_calls": "lower",
    }
    rh_scores, gr_scores = [], []
    for m in metrics:
        a = float(final_rh[m])
        b = float(final_gr[m])
        if directions[m] == "higher":
            denom = max(abs(a), abs(b), 1e-12)
            rh_scores.append(a / denom)
            gr_scores.append(b / denom)
        else:
            positive = [v for v in [a, b] if v > 1e-12]
            best = min(positive) if positive else 1.0
            rh_scores.append(best / max(a, 1e-12))
            gr_scores.append(best / max(b, 1e-12))

    x = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, rh_scores, width, label="RH-NBV")
    plt.bar(x + width / 2, gr_scores, width, label="GradientNBV")
    plt.xticks(x, metrics, rotation=25, ha="right")
    plt.ylim(0, 1.15)
    plt.ylabel("Normalized score; 1.0 = better final result")
    plt.title("Final normalized comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "final_normalized_summary.png")
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    return path


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Using torch.cross without specifying the dim arg is deprecated.*")

    rospy.init_node("comparison_test", log_level=rospy.WARN)
    os.makedirs("results", exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    occ = detect_occlusion_type()
    print_title(occ)

    rh_vp, rh_rows = run_method("RH-NBV")
    rh_results = collect_results(rh_vp, "RH-NBV", occ)
    save_and_print(rh_results, prefix="results/results")

    grad_vp, grad_rows = run_method("GradientNBV")
    grad_results = collect_results(grad_vp, "GradientNBV", occ)
    save_and_print(grad_results, prefix="results/results")

    all_rows = rh_rows + grad_rows
    save_live_csv(all_rows)
    save_existing_style_plots(rh_vp, grad_vp, rh_rows, grad_rows, occ)
    print_final_comparison(rh_rows, grad_rows)

    print("\nDone. JSON files are in results/. CSV and plots are in results/.")
