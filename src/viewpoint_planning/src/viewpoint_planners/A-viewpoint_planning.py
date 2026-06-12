import os
import math
import torch
import time
import xml.etree.ElementTree as ET

import numpy as np
import rospy
from scipy.spatial import KDTree

from abb_control.arm_control_client import ArmControlClient
from perception.perceiver import Perceiver
from viewpoint_planners.rh_planner import RHPlanner
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from utils.py_utils import numpy_to_pose
from utils.sdf_spawner import SDFSpawner


class ViewpointPlanning:

    def __init__(self, lr=None, trial_seed=None, start_jitter=0.03,
                 rh_params=None):
        # ROS 
        self.arm_control = ArmControlClient()
        self.perceiver = Perceiver()
        self.viewpoint_sampler = ViewpointSampler()
        self.sdf_spawner = SDFSpawner()
        
        self.lr = lr
        # Per-trial reproducible start variation (Burusa uses +-3cm ROI
        # uncertainty across trials). None => deterministic predefined start.
        self.trial_seed = trial_seed
        self.start_jitter = start_jitter

        # RH hyperparameters — overridable for ablation studies (H, K, gamma).
        # Defaults match the thesis baseline configuration.
        default_rh = {
            "horizon": 3,            # H
            "num_candidates": 10,    # K
            "lambda_cost": 2.0,      # lambda
            "step_size": 0.065,
            "bias_ratio": 0.7,
            "discount": 0.85,        # gamma
            "r_min": 0.15,
            "r_max": 0.45,
            "occlusion_bonus": 2.0,
            "stagnation_patience": 4,
            "use_spherical_bounds": True,   # False => Burusa-style box-only
        }
        if rh_params:
            default_rh.update(rh_params)
        self.rh_params = default_rh

        # Target node centred on the bunny body. This point sits on the
        # camera-facing surface (the camera starts near Y=-0.05 and looks toward
        # -Y), so the ROI around it is actually observable. Empirically this is
        # the value that yields non-zero coverage; moving the target to the
        # densest mesh region instead put it on the occluded far/corner side and
        # drove coverage to 0. Overridable via TARGET_POS env for node ablation.
        self.target_position = np.array([0.5, -0.4, 1.1])
        _tp_env = os.environ.get("TARGET_POS")
        if _tp_env:
            self.target_position = np.array([float(v) for v in _tp_env.split(",")])
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
            use_spherical_bounds=self.rh_params["use_spherical_bounds"],
            # Robot kinematic reach: measured from arm failure positions
            # x:[0.30,0.70]  y:[-0.25,0.15]  z:[0.95,1.30]
            robot_reach_bounds=np.array([[0.30,-0.15,0.97],[0.65,0.05,1.25]]),
        )
        init_time_rh = time.time() - start_time

        # metric arrays
        self.losses_rh = np.array([0.0])
        self.cumulative_time_rh = np.array([init_time_rh])
        self.coverages_rh = np.array([0.0])
        self.trail_rh = [self.camera_pose[:3].copy()]
        self.trajectory_distance_rh = np.array([0.0])
        self.recall_rh = np.array([0.0])
        self.precision_rh = np.array([0.0])
        self.f1_rh = np.array([0.0])
        self.ray_calls_rh = np.array([0])
        self.sigma_rh = np.array([0.0])
        self.occluded_recall_rh = np.array([0.0])
        # Explicit TP/FP/FN series (Burusa-style, supervisor request)
        self.tp_rh = np.array([0])
        self.fp_rh = np.array([0])
        self.fn_rh = np.array([0])
        # When True, calculate_F1 prints the voxel->mesh distance distribution.
        # The driver toggles this on for the final iteration.
        self._diagnose_f1 = False

        print(
            f"[RH] K={self.rh_planner.num_candidates}, "
            f"H={self.rh_planner.horizon} -> "
            f"evals/iter={self.rh_planner.num_candidates * self.rh_planner.horizon}"
        )

    def config(self):
        self.spawn_no_occlusion()
        #self.spawn_easy_occlusion()
        # self.spawn_hard_occlusion()
        # self.spawn_extreme_occlusion()
        #self.spawn_complex_occlusion()

        self.camera_pose = self.viewpoint_sampler.predefine_start_pose(
            self.target_position
        )

        # Per-trial start variation: reproducible +-start_jitter on the camera
        # position only (orientation is recomputed by look_at toward target).
        # Mirrors Burusa's +-3cm ROI/start uncertainty across trials so that a
        # single run is comparable to their 288-trial average.
        if self.trial_seed is not None:
            rng = np.random.default_rng(self.trial_seed)
            jitter = rng.uniform(-self.start_jitter, self.start_jitter, size=3)
            self.camera_pose[:3] = self.camera_pose[:3] + jitter

        if self.arm_control:
            self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))

        self.grid_size = np.array([0.3, 0.6, 0.3])
        self.grid_center = self.target_position

        camera_info = self.perceiver.get_camera_info()
        self.image_size = np.array([camera_info.width, camera_info.height])
        self.intrinsics = np.array(camera_info.K).reshape(3, 3)

    # -------------------------------------------------------------
    # Occlusion scenarios
    # ---------------------------------------------------------------
    def spawn_no_occlusion(self):
        """No occluding object is spawned."""
        pass

    def spawn_easy_occlusion(self):
        """Single box, offset to the side."""
        self.sdf_spawner.spawn_box(np.array([0.65, -0.3, 1.1]), 1)

    def spawn_hard_occlusion(self):
        """Single box, closer to the target and more centered."""
        self.sdf_spawner.spawn_box(np.array([0.6, -0.25, 1.1]), 1)

    def spawn_extreme_occlusion(self):
        """Two stacked boxes aligned in front of the target."""
        self.sdf_spawner.spawn_box(np.array([0.6, -0.3, 1.1]), 1)
        self.sdf_spawner.spawn_box(np.array([0.6, -0.3, 1.2]), 2)

    def spawn_complex_occlusion(self):
        """Three-object occlusion setup."""
        self.sdf_spawner.spawn_box(np.array([0.73, -0.25, 0.95]), 1)
        self.sdf_spawner.spawn_bar(np.array([0.5, -0.22, 1.0]), 2)
        self.sdf_spawner.spawn_box(np.array([0.6, -0.32, 1.3]), 3)

    # -----------------------------------------------------
    # RH execution
    # -----------------------------------------------------
    def run_rh(self):
        """Run one Receding Horizon NBV iteration and log RH-only metrics."""
        start_time = time.time()
        current_coverage = float(self.coverages_rh[-1]) if len(self.coverages_rh) > 0 else 0.0
        self.camera_pose, loss, n_evals = self.rh_planner.rh_view(current_coverage=current_coverage)

        self.camera_pose[:3] = self._clamp_to_rh_bounds(self.camera_pose[:3])
        self.trail_rh.append(self.camera_pose[:3].copy())
        self.losses_rh = np.append(self.losses_rh, loss)

        is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        rospy.sleep(1.0)

        if is_success:
            depth_image, _, semantics = self.perceiver.run()
            coverage = self.rh_planner.update_voxel_grid(
                depth_image, semantics, self.camera_pose
            )
            if self.rh_planner.occluded_mesh_points is None:
                self.rh_planner.set_occluded_mesh_points()
            self.coverages_rh = np.append(self.coverages_rh, coverage)
            self.trajectory_distance_rh = np.append(
                self.trajectory_distance_rh,
                self.trajectory_distance_rh[-1] + self._last_step_distance(),
            )
        else:
            # Auto-shrink reach bounds based on the failed position
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
            # Reset to last good pose
            if len(self.trail_rh) >= 2:
                self.rh_planner.current_pos = torch.tensor(
                    self.trail_rh[-2], dtype=torch.float32,
                    device=self.rh_planner.device
                )
            coverage = self.coverages_rh[-1]
            self.coverages_rh = np.append(self.coverages_rh, coverage)
            self.trajectory_distance_rh = np.append(
                self.trajectory_distance_rh, self.trajectory_distance_rh[-1]
            )

        iter_time = time.time() - start_time
        self.cumulative_time_rh = np.append(
            self.cumulative_time_rh, self.cumulative_time_rh[-1] + iter_time
        )

        f1, recall, precision = self.rh_planner.calculate_F1(diagnose=self._diagnose_f1)
        self.f1_rh = np.append(self.f1_rh, f1)
        self.recall_rh = np.append(self.recall_rh, recall)
        self.precision_rh = np.append(self.precision_rh, precision)
        # Log explicit TP/FP/FN counts from the F1 computation
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
        """Return all collected RH-only metrics."""
        return {
            "losses": self.losses_rh,
            "cumulative_time": self.cumulative_time_rh,
            "coverages": self.coverages_rh,
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


    # ----------Mesh loading-------------
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
        # Z translation of the ground-truth mesh. This MUST be calibrated so the
        # mesh coincides with the bunny the depth camera actually reconstructs.
        # Gazebo spawns the bunny at z=1.0 (bunny.world). The correct vertical
        # correction is being determined empirically via the F1 DIAG output;
        # override with env var MESH_Z_CORR (metres subtracted from 1.0).
        import os as _os
        z_corr = float(_os.environ.get("MESH_Z_CORR", 0.048))
        translation = np.array([0.5, -0.4, 1.0 - z_corr])
        transformed_coords = vertices_swapped * scale + translation
        mesh_tree = KDTree(transformed_coords)
        return transformed_coords, mesh_tree
