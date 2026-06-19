"""
planner_eval_mixin.py — shared evaluation code for ALL baseline planners.

RH-NBV defines the metric logic (F1, sigma, occluded recall) in rh_planner.py.
For a fair comparison every baseline (GradientNBV, PSO, Random) must score
reconstruction with the EXACT same code — same ROI crop, same threshold, same
class filter, same occluder masking, same TP/FP/FN convention. Copy-pasting
that ~100-line block into each planner invites drift, so it lives here once and
each baseline mixes it in.

This is MEASUREMENT only. It never touches how a planner CHOOSES viewpoints,
so each planner keeps its own strategy (gradient ascent / PSO swarm / random
sampling). The mixin assumes the host class provides:
    self.voxel_grid        (with .get_occupied_points(), .voxel_size)
    self.target_params     (torch tensor or array, the ROI centre)
    self.mesh_coordinates  ((M,3) ground-truth mesh)
and that the host sets these attributes in __init__:
    self.target_voxels = np.array(0)
    self.all_target_voxels = np.zeros((0, 3))
    self.occluded_mesh_points = None
    self.last_tp = self.last_fp = self.last_fn = 0
Use init_eval_state(self) to set them all at once.
"""
import os
import numpy as np
from scipy.spatial import KDTree


def init_eval_state(planner):
    """Initialise the evaluation attributes the mixin relies on."""
    planner.target_voxels = np.array(0)
    planner.all_target_voxels = np.zeros((0, 3))
    planner.occluded_mesh_points = None
    planner.last_tp = 0
    planner.last_fp = 0
    planner.last_fn = 0


class PlannerEvalMixin:
    """Evaluation helpers identical to RHPlanner, shared across baselines."""

    # ---- Occupied voxel access (identical to RH/GradientNBV) ----
    def get_occupied_points(self):
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        return (
            voxel_points.cpu().numpy(),
            sem_conf_scores.cpu().numpy(),
            sem_class_ids.cpu().numpy(),
        )

    # ---- F1 / recall / precision — Burusa Table II aligned, RH-identical ----
    def calculate_F1(self, occluder_positions=None, match_threshold=None,
                     diagnose=False):
        """3D node reconstruction F1, identical logic to RHPlanner.calculate_F1.
        Target-class voxels only, occluder masking, ROI crop (ROI_HALF), and a
        resolution-scaled match threshold (F1_THRESH default 4x voxel size).
        Sets self.last_tp/fp/fn and returns (f1, recall, precision)."""
        voxel_points, _, sem_class = self.get_occupied_points()

        self.last_tp = 0
        self.last_fp = 0
        self.last_fn = 0

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        # 1) Keep only target-class voxels (Burusa: class 0 = fruit node).
        sem_class = np.asarray(sem_class)
        if sem_class.shape[0] == voxel_points.shape[0]:
            voxel_points = voxel_points[sem_class == 0]

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        # 2) Mask out known occluder voxels.
        if occluder_positions:
            keep = np.ones(len(voxel_points), dtype=bool)
            for center, half in occluder_positions:
                in_occ = np.all(
                    np.abs(voxel_points - np.array(center)) <= np.array(half),
                    axis=1,
                )
                keep &= ~in_occ
            voxel_points = voxel_points[keep]

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        # Full reconstruction (pre-ROI) for the reconstruction plot.
        self.all_target_voxels = voxel_points.copy()

        # 3) Clip voxels and mesh to the ROI cube around the target.
        target = self.target_params
        if hasattr(target, "detach"):
            target = target.detach().cpu().numpy()
        target = np.asarray(target)
        roi_half = float(os.environ.get("ROI_HALF", 0.075))
        voxel_points = voxel_points[
            np.all(np.abs(voxel_points - target) <= roi_half, axis=1)
        ]
        roi_mesh = self.mesh_coordinates[
            np.all(np.abs(self.mesh_coordinates - target) <= roi_half, axis=1)
        ]

        if len(voxel_points) == 0 or len(roi_mesh) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        self.target_voxels = voxel_points
        mesh_tree = KDTree(roi_mesh)
        voxel_tree = KDTree(voxel_points)

        # Threshold: voxel-resolution-scaled (default 4x voxel ~12mm) because
        # occupancy-thresholded voxel centres sit ~9-13mm off the mesh surface.
        if match_threshold is None:
            vs = self.voxel_grid.voxel_size
            if hasattr(vs, "detach"):
                vs = vs.detach().cpu().numpy()
            vsize = float(np.asarray(vs).reshape(-1)[0])
            match_threshold = float(os.environ.get("F1_THRESH", vsize * 4.0))
        half = match_threshold
        radius = half * np.sqrt(3)

        if diagnose:
            d, _ = mesh_tree.query(voxel_points)
            pct = np.percentile(d, [0, 25, 50, 75, 100]) * 1000
            print(f"  [F1 DIAG][{type(self).__name__}] voxel->mesh nearest (mm): "
                  f"min={pct[0]:.1f} med={pct[2]:.1f} max={pct[4]:.1f} | "
                  f"thr={half*1000:.1f}mm | within thr="
                  f"{float(np.mean(d <= half))*100:.1f}%")

        nr_correct = 0
        for v in voxel_points:
            for idx in mesh_tree.query_ball_point(v, r=radius):
                if all(abs(v[d_] - roi_mesh[idx][d_]) <= half for d_ in range(3)):
                    nr_correct += 1
                    break

        nr_recalled = 0
        for c in roi_mesh:
            for idx in voxel_tree.query_ball_point(c, r=radius):
                if all(abs(voxel_points[idx][d_] - c[d_]) <= half for d_ in range(3)):
                    nr_recalled += 1
                    break

        self.last_tp = nr_correct
        self.last_fp = len(voxel_points) - nr_correct
        self.last_fn = len(roi_mesh) - nr_recalled

        precision = nr_correct / len(voxel_points)
        recall = nr_recalled / len(roi_mesh)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0)
        return f1, recall, precision

    # ---- Occluded-recall baseline + recall (identical to RH) ----
    def set_occluded_mesh_points(self):
        voxel_points, _, _ = self.get_occupied_points()
        vs = self.voxel_grid.voxel_size
        if hasattr(vs, "detach"):
            vs = vs.detach().cpu().numpy()
        half = float(np.asarray(vs).reshape(-1)[0]) * 4.0
        radius = half * np.sqrt(3)
        if len(voxel_points) == 0:
            self.occluded_mesh_points = self.mesh_coordinates.copy()
            return
        voxel_tree = KDTree(voxel_points)
        unseen = []
        for coord in self.mesh_coordinates:
            idxs = voxel_tree.query_ball_point(coord, r=radius)
            covered = any(
                all(abs(voxel_points[i][d] - coord[d]) <= half for d in range(3))
                for i in idxs
            )
            if not covered:
                unseen.append(coord)
        self.occluded_mesh_points = np.array(unseen) if unseen else np.zeros((0, 3))
        print(f"[{type(self).__name__}] Occluded after view 0: "
              f"{len(self.occluded_mesh_points)}/{len(self.mesh_coordinates)} "
              f"({100*len(self.occluded_mesh_points)/len(self.mesh_coordinates):.1f}%)")

    def compute_occluded_recall(self) -> float:
        if self.occluded_mesh_points is None or len(self.occluded_mesh_points) == 0:
            return 0.0
        voxel_points, _, _ = self.get_occupied_points()
        if len(voxel_points) == 0:
            return 0.0
        voxel_tree = KDTree(voxel_points)
        vs = self.voxel_grid.voxel_size
        if hasattr(vs, "detach"):
            vs = vs.detach().cpu().numpy()
        half = float(np.asarray(vs).reshape(-1)[0]) * 4.0
        radius = half * np.sqrt(3)
        recovered = 0
        for coord in self.occluded_mesh_points:
            idxs = voxel_tree.query_ball_point(coord, r=radius)
            if any(
                all(abs(voxel_points[i][d] - coord[d]) <= half for d in range(3))
                for i in idxs
            ):
                recovered += 1
        return recovered / len(self.occluded_mesh_points)

    # ---- Sigma: spatial spread of detected target voxels (identical to RH) ----
    def compute_sigma(self) -> float:
        if (not isinstance(self.target_voxels, np.ndarray)
                or self.target_voxels.ndim < 2):
            return 0.0
        if len(self.target_voxels) == 0:
            return 0.0
        centroid = self.target_voxels.mean(axis=0)
        dists = np.linalg.norm(self.target_voxels - centroid, axis=1)
        return float(dists.mean())
