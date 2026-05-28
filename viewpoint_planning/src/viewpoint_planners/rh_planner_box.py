"""
rh_planner_box.py
-----------------
Thin subclass of RHPlanner for the simple box sanity-check experiment.

WHY A SEPARATE FILE?
--------------------
rh_planner.py hardcodes  `half = 0.002`  (2 mm) in three places:
  - calculate_F1()
  - set_occluded_mesh_points()
  - compute_occluded_recall()

This works for the Stanford Bunny (mesh vertices ~2 mm apart after scaling).
For the box we use voxel_size=0.008m and mesh spacing ~7.9mm, so the
2mm tolerance means voxels and mesh points never match → F1 stays near 0.

FIX: override those three methods with BOX_MESH_HALF = 0.008 m.
Everything else in RHPlanner is unchanged.
"""

import numpy as np
from scipy.spatial import KDTree

from viewpoint_planners.rh_planner import RHPlanner


BOX_MESH_HALF = 0.008   # matches voxel_size=0.008 and mesh spacing ~7.9mm


class RHPlannerBox(RHPlanner):
    """RHPlanner subclass for the box experiment — corrected match tolerance."""

    # ------------------------------------------------------------------
    # set_occluded_mesh_points
    # ------------------------------------------------------------------
    def set_occluded_mesh_points(self):
        """Same as parent but uses BOX_MESH_HALF tolerance (was 0.002)."""
        voxel_points, _, _ = self.get_occupied_points()
        half   = BOX_MESH_HALF
        radius = half * np.sqrt(3)
        if len(voxel_points) == 0:
            self.occluded_mesh_points = self.mesh_coordinates.copy()
            return
        voxel_tree = KDTree(voxel_points)
        unseen = []
        for coord in self.mesh_coordinates:
            idxs    = voxel_tree.query_ball_point(coord, r=radius)
            covered = any(
                all(abs(voxel_points[i][d] - coord[d]) <= half for d in range(3))
                for i in idxs
            )
            if not covered:
                unseen.append(coord)
        self.occluded_mesh_points = np.array(unseen) if unseen else np.zeros((0, 3))
        print(
            f"[RHPlannerBox] Occluded after view 0: "
            f"{len(self.occluded_mesh_points)}/{len(self.mesh_coordinates)} "
            f"({100*len(self.occluded_mesh_points)/len(self.mesh_coordinates):.1f}%)"
        )

    # ------------------------------------------------------------------
    # compute_occluded_recall
    # ------------------------------------------------------------------
    def compute_occluded_recall(self) -> float:
        """Same as parent but uses BOX_MESH_HALF tolerance (was 0.002)."""
        if self.occluded_mesh_points is None or len(self.occluded_mesh_points) == 0:
            return 0.0
        voxel_points, _, _ = self.get_occupied_points()
        if len(voxel_points) == 0:
            return 0.0
        voxel_tree = KDTree(voxel_points)
        half       = BOX_MESH_HALF
        radius     = half * np.sqrt(3)
        recovered  = 0
        for coord in self.occluded_mesh_points:
            idxs = voxel_tree.query_ball_point(coord, r=radius)
            if any(
                all(abs(voxel_points[i][d] - coord[d]) <= half for d in range(3))
                for i in idxs
            ):
                recovered += 1
        return recovered / len(self.occluded_mesh_points)

    # ------------------------------------------------------------------
    # calculate_F1
    # ------------------------------------------------------------------
    def calculate_F1(self, occluder_positions=None):
        """Same as parent but uses BOX_MESH_HALF tolerance (was 0.002)."""
        voxel_points, _, _ = self.get_occupied_points()
        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0.0, 0.0, 0.0

        if occluder_positions:
            keep = np.ones(len(voxel_points), dtype=bool)
            for center, half_size in occluder_positions:
                c = np.array(center)
                h = np.array(half_size)
                in_occ = np.all(np.abs(voxel_points - c) <= h, axis=1)
                keep &= ~in_occ
            voxel_points = voxel_points[keep]

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0.0, 0.0, 0.0

        self.target_voxels = voxel_points
        mesh_tree  = KDTree(self.mesh_coordinates)
        voxel_tree = KDTree(voxel_points)
        half   = BOX_MESH_HALF
        radius = half * np.sqrt(3)

        nr_correct = 0
        for voxel in voxel_points:
            for idx in mesh_tree.query_ball_point(voxel, r=radius):
                coord = self.mesh_coordinates[idx]
                if all(abs(voxel[d] - coord[d]) <= half for d in range(3)):
                    nr_correct += 1
                    break

        nr_recalled = 0
        for coord in self.mesh_coordinates:
            for idx in voxel_tree.query_ball_point(coord, r=radius):
                voxel = voxel_points[idx]
                if all(abs(voxel[d] - coord[d]) <= half for d in range(3)):
                    nr_recalled += 1
                    break

        precision = nr_correct  / len(voxel_points)
        recall    = nr_recalled / len(self.mesh_coordinates)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        return f1, recall, precision
