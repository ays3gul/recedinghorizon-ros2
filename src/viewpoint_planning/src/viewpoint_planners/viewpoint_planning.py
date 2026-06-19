import os
import math
import torch
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import KDTree

from abb_control.arm_control_client import ArmControlClient
from perception.perceiver import Perceiver
from viewpoint_planners.rh_planner import RHPlanner
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from utils.py_utils import numpy_to_pose
from utils.sdf_spawner import SDFSpawner
from viewpoint_planners.fair_comparison_config import (
    reach_bounds_for_start,
    get_target_position as fc_get_target_position,
    GRID_SIZE as FC_GRID_SIZE,
)


class ViewpointPlanning:

    def __init__(self, lr=None, trial_seed=None, start_jitter=0.03,
                 rh_params=None):
        # ROS 2
        self.arm_control = ArmControlClient()
        self.perceiver = Perceiver()
        self.viewpoint_sampler = ViewpointSampler()
        self.sdf_spawner = SDFSpawner()

        self.lr = lr
        self.trial_seed = trial_seed
        self.start_jitter = start_jitter

        default_rh = {
            "horizon": 3,
            "num_candidates": 10,
            "lambda_cost": 2.0,
            "step_size": 0.065,
            "bias_ratio": 0.7,
            "discount": 0.85,
            "r_min": 0.15,
            "r_max": 0.45,
            "occlusion_bonus": 2.0,
            "stagnation_patience": 4,
            "stagnation_threshold": 1.5,
            "use_spherical_bounds": True,
        }
        if rh_params:
            default_rh.update(rh_params)
        self.rh_params = default_rh

        self.target_position = fc_get_target_position()
        self.config()
        self.mesh_coordinates, self.mesh_tree = self.get_mesh_coordinates()

        start_time = time.time()
        self.rh_planner = RHPlanner(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=1,
            mesh_coordinates=self.mesh_coordinates,
            mesh_tree=self.mesh_tree,
            horizon=self.rh_params["horizon"],
            num_candidates=self.rh_params["num_candidates"],
            lambda_cost=self.rh_params["lambda_cost"],
            step_size=self.rh_params["step_size"],
            bias_ratio=self.rh_params["bias_ratio"],
            discount=self.rh_params["discount"],
            r_min=self.rh_params["r_min"],
            r_max=self.rh_params["r_max"],
            occlusion_bonus=self.rh_params["occlusion_bonus"],
            stagnation_patience=self.rh_params["stagnation_patience"],
            stagnation_threshold=self.rh_params.get("stagnation_threshold", 1.5),
            use_spherical_bounds=self.rh_params["use_spherical_bounds"],
            robot_reach_bounds=reach_bounds_for_start(self.camera_pose),
        )
        init_time_rh = time.time() - start_time

        self.losses_rh = np.array([0.0])
        self.cumulative_time_rh = np.array([init_time_rh])
        self.coverages_rh = np.array([0.0])
        self.voxels_seen_rh = np.array([0])
        self.voxels_total_rh = np.array([0])
        self.trail_rh = [self.camera_pose[:3].copy()]
        self.trajectory_distance_rh = np.array([0.0])
        self.recall_rh = np.array([0.0])
        self.precision_rh = np.array([0.0])
        self.f1_rh = np.array([0.0])
        self.ray_calls_rh = np.array([0])
        self.sigma_rh = np.array([0.0])
        self.occluded_recall_rh = np.array([0.0])
        self.tp_rh = np.array([0])
        self.fp_rh = np.array([0])
        self.fn_rh = np.array([0])
        self._diagnose_f1 = False

        # Set occluded baseline on the empty grid (before any planning) so
        # that every planner starts from an identical 100% occluded baseline.
        self.rh_planner.set_occluded_mesh_points()

        print(
            f"[RH] K={self.rh_planner.num_candidates}, "
            f"H={self.rh_planner.horizon} -> "
            f"evals/iter={self.rh_planner.num_candidates * self.rh_planner.horizon}"
        )

    def config(self):
        occ = os.environ.get("OCC", "none").lower()
        self._occ_type = occ
        spawn_fn = {
            "none":     self.spawn_no_occlusion,
            "frontal":  self.spawn_frontal_occlusion,
            "half_box": self.spawn_half_box_occlusion,
            "tunnel":   self.spawn_tunnel_occlusion,
            "well":     self.spawn_well_occlusion,
        }.get(occ, self.spawn_no_occlusion)
        print(f"[Occlusion] scenario = {occ}")
        spawn_fn()

        self.camera_pose = self.viewpoint_sampler.predefine_start_pose(
            self.target_position
        )

        if self.trial_seed is not None:
            rng = np.random.default_rng(self.trial_seed)
            jitter = rng.uniform(-self.start_jitter, self.start_jitter, size=3)
            self.camera_pose[:3] = self.camera_pose[:3] + jitter

        if self.arm_control:
            self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))

        self.grid_size = np.array(FC_GRID_SIZE, dtype=float)
        self.grid_center = self.target_position

        camera_info = self.perceiver.get_camera_info()
        if camera_info is None:
            raise RuntimeError(
                "[ViewpointPlanning] Camera info not received within timeout. "
                "Check that the simulation is running and the ros_gz_bridge camera topics are being published."
            )
        self.image_size = np.array([camera_info.width, camera_info.height])
        self.intrinsics = np.array(camera_info.k).reshape(3, 3)  # ROS 2: lowercase k

    # -------------------------------------------------------------
    # Occlusion scenarios
    # -------------------------------------------------------------
    def spawn_no_occlusion(self):
        pass

    def spawn_frontal_occlusion(self):
        self.sdf_spawner.spawn_named_model(np.array([0.5, -0.15, 1.12]), 1, "panel_front")

    def spawn_half_box_occlusion(self):
        self.sdf_spawner.spawn_named_model(np.array([0.40, -0.25, 1.10]), 1, "panel_side")
        self.sdf_spawner.spawn_named_model(np.array([0.64, -0.25, 1.10]), 2, "panel_side")
        self.sdf_spawner.spawn_named_model(np.array([0.50, -0.36, 1.10]), 3, "panel_back")

    def spawn_tunnel_occlusion(self):
        self.sdf_spawner.spawn_named_model(np.array([0.43, -0.25, 1.10]), 1, "panel_tunnel")
        self.sdf_spawner.spawn_named_model(np.array([0.57, -0.25, 1.10]), 2, "panel_tunnel")

    def spawn_well_occlusion(self):
        self.sdf_spawner.spawn_named_model(np.array([0.40, -0.25, 1.08]), 1, "panel_side_low")
        self.sdf_spawner.spawn_named_model(np.array([0.64, -0.25, 1.08]), 2, "panel_side_low")
        self.sdf_spawner.spawn_named_model(np.array([0.52, -0.13, 1.08]), 3, "panel_front_low")
        self.sdf_spawner.spawn_named_model(np.array([0.52, -0.37, 1.08]), 4, "panel_front_low")

    # -------------------------------------------------------------
    # RH execution
    # -------------------------------------------------------------
    def run_rh(self):
        """Run one Receding Horizon NBV iteration and log RH-only metrics."""
        start_time = time.time()
        current_coverage = float(self.coverages_rh[-1]) if len(self.coverages_rh) > 0 else 0.0
        self.camera_pose, loss, n_evals = self.rh_planner.rh_view(current_coverage=current_coverage)

        self.camera_pose[:3] = self._clamp_to_rh_bounds(self.camera_pose[:3])
        self.trail_rh.append(self.camera_pose[:3].copy())
        self.losses_rh = np.append(self.losses_rh, loss)

        is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        time.sleep(1.0)  # ROS 2: time.sleep instead of rospy.sleep

        if is_success:
            depth_image, _, semantics = self.perceiver.run()
            if depth_image is not None and semantics is not None:
                coverage = self.rh_planner.update_voxel_grid(
                    depth_image, semantics, self.camera_pose
                )
            else:
                coverage = self.coverages_rh[-1] if len(self.coverages_rh) > 0 else 0.0
            self.coverages_rh = np.append(self.coverages_rh, coverage)
            self.voxels_seen_rh = np.append(self.voxels_seen_rh, self.rh_planner.voxel_grid.n_seen)
            self.voxels_total_rh = np.append(self.voxels_total_rh, self.rh_planner.voxel_grid.n_total)
            self.rh_planner.visualize()
            self.trajectory_distance_rh = np.append(
                self.trajectory_distance_rh,
                self.trajectory_distance_rh[-1] + self._last_step_distance(),
            )
        else:
            failed_pos = self.camera_pose[:3]
            if self.rh_planner.robot_reach_bounds is not None:
                bounds = self.rh_planner.robot_reach_bounds.cpu().numpy().copy()
                for dim in range(3):
                    if failed_pos[dim] < bounds[0][dim] + 0.02:
                        bounds[0][dim] = min(failed_pos[dim] + 0.03, bounds[1][dim] - 0.05)
                    if failed_pos[dim] > bounds[1][dim] - 0.02:
                        bounds[1][dim] = max(failed_pos[dim] - 0.03, bounds[0][dim] + 0.05)
                self.rh_planner.robot_reach_bounds = torch.tensor(
                    bounds, dtype=torch.float32, device=self.rh_planner.device
                )
                print(f'[VP] Reach bounds tightened: y=[{bounds[0][1]:.3f},{bounds[1][1]:.3f}]')
            if len(self.trail_rh) >= 2:
                self.rh_planner.current_pos = torch.tensor(
                    self.trail_rh[-2], dtype=torch.float32,
                    device=self.rh_planner.device
                )
            coverage = self.coverages_rh[-1]
            self.coverages_rh = np.append(self.coverages_rh, coverage)
            self.voxels_seen_rh = np.append(self.voxels_seen_rh, self.voxels_seen_rh[-1])
            self.voxels_total_rh = np.append(self.voxels_total_rh, self.voxels_total_rh[-1])
            self.trajectory_distance_rh = np.append(
                self.trajectory_distance_rh, self.trajectory_distance_rh[-1]
            )

        iter_time = time.time() - start_time
        self.cumulative_time_rh = np.append(
            self.cumulative_time_rh, self.cumulative_time_rh[-1] + iter_time
        )

        occ_positions = None
        if getattr(self, "_occ_type", "none") == "tunnel":
            occ_positions = [
                (np.array([0.43, -0.25, 1.10]), np.array([0.012, 0.052, 0.102])),
                (np.array([0.57, -0.25, 1.10]), np.array([0.012, 0.052, 0.102])),
            ]
        f1, recall, precision = self.rh_planner.calculate_F1(
            occluder_positions=occ_positions, diagnose=self._diagnose_f1)
        self.f1_rh = np.append(self.f1_rh, f1)
        self.recall_rh = np.append(self.recall_rh, recall)
        self.precision_rh = np.append(self.precision_rh, precision)
        self.tp_rh = np.append(self.tp_rh, self.rh_planner.last_tp)
        self.fp_rh = np.append(self.fp_rh, self.rh_planner.last_fp)
        self.fn_rh = np.append(self.fn_rh, self.rh_planner.last_fn)
        self.ray_calls_rh = np.append(self.ray_calls_rh, self.rh_planner.ray_trace_count)
        sigma = self.rh_planner.compute_sigma()
        self.sigma_rh = np.append(self.sigma_rh, sigma)
        occ_recall = self.rh_planner.compute_occluded_recall()
        self.occluded_recall_rh = np.append(self.occluded_recall_rh, occ_recall)

        print(
            f"[RH] coverage={coverage:.4f} | loss={loss:.4f} | "
            f"F1={f1:.4f} | recall={recall:.4f} | "
            f"precision={precision:.4f} | occ_recall={occ_recall:.4f} | evals={n_evals}"
        )
        total_occluded = len(self.rh_planner.occluded_mesh_points) \
            if self.rh_planner.occluded_mesh_points is not None else 0
        recovered = int(occ_recall * total_occluded) if total_occluded > 0 else 0
        print(
            f"    Occluded recall: {occ_recall*100:.1f}% "
            f"({recovered}/{total_occluded} mesh points recovered)"
        )

        return coverage, loss, f1, recall, precision, n_evals

    # HELPERS
    def _clamp_to_rh_bounds(self, position):
        bounds = self.rh_planner.camera_bounds.detach().cpu().numpy()
        return np.clip(position, bounds[0], bounds[1])

    def _last_step_distance(self):
        p1 = self.trail_rh[-2]
        p2 = self.trail_rh[-1]
        return math.sqrt(
            (p2[0] - p1[0]) ** 2
            + (p2[1] - p1[1]) ** 2
            + (p2[2] - p1[2]) ** 2
        )

    def get_rh_metrics(self):
        return {
            "losses": self.losses_rh,
            "cumulative_time": self.cumulative_time_rh,
            "coverages": self.coverages_rh,
            "voxels_seen": self.voxels_seen_rh,
            "voxels_total": self.voxels_total_rh,
            "trajectory_distance": self.trajectory_distance_rh,
            "f1": self.f1_rh,
            "recall": self.recall_rh,
            "precision": self.precision_rh,
            "ray_calls": self.ray_calls_rh,
            "trail": np.array(self.trail_rh),
            "sigma": self.sigma_rh,
            "occluded_recall": self.occluded_recall_rh,
            "tp": self.tp_rh,
            "fp": self.fp_rh,
            "fn": self.fn_rh,
        }

    # ---------- Mesh loading ----------
    def get_mesh_coordinates(self):
        file_path = "/home/ayse/Desktop/RecedingHorizon/src/simulation_environment/meshes/bunny.dae"
        tree = ET.parse(file_path)
        root = tree.getroot()
        namespaces = {"ns": "http://www.collada.org/2005/11/COLLADASchema"}
        positions_array = root.find(
            ".//ns:float_array[@id='bun_zipper-mesh-positions-array']", namespaces
        )
        if positions_array is None:
            raise ValueError("Positions array not found in the COLLADA file.")
        raw_data = list(map(float, positions_array.text.split()))
        vertices = np.array(raw_data).reshape(-1, 3)
        vertices_swapped = vertices[:, [0, 2, 1]]
        scale = np.array([-1.2, 1.2, 1.2])
        z_corr = float(os.environ.get("MESH_Z_CORR", 0.0))
        translation = np.array([0.5, -0.25, 1.0 - z_corr])
        transformed_coords = vertices_swapped * scale + translation
        mesh_tree = KDTree(transformed_coords)
        return transformed_coords, mesh_tree
