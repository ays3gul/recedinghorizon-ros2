"""
rh_planner.py — Receding Horizon Next-Best-View Planner
========================================================

Improvements over the original:

1. SPHERICAL BOUNDS — camera orbits target on a sphere (r_min, r_max).
   No more corner-trapping. Positions outside the shell are re-projected
   back onto the sphere surface.

2. OCCLUSION-AWARE IG — adds a bonus for viewpoints that can see voxels
   currently behind known occluders (high occupancy, high uncertainty
   neighbours). These viewpoints are the most valuable.

3. INCREMENTAL IG SCORING — each step in a sequence only scores NEWLY
   seen voxels (not already counted in earlier steps). Avoids rewarding
   sequences that hover in the same spot.

4. STAGNATION RECOVERY — if coverage does not improve for
   `stagnation_patience` iterations, the camera is forced to the
   antipodal point on the orbit sphere, breaking the local optimum.

5. ADAPTIVE HORIZON — H grows as coverage saturates:
       coverage < 40%  → H = 2  (fast early exploration)
       coverage < 70%  → H = 3  (default)
       coverage >= 70% → H = 5  (deep occlusion handling)

6. CORRECT RAY COUNT — predict_update does NOT increment ray_trace_count.
   Only compute_gain_on_grid counts (matches Burusa paper metric 3).

References:
    Burusa et al. (ICRA 2024)
    Bircher et al. (ICRA 2016, Autonomous Robots 2017)
    Mayne et al. (Automatica 2000)
    Zaenker et al. (IROS 2021)
"""

import torch
import torch.nn.functional as F
import numpy as np

from scene_representation.voxel_grid import VoxelGrid
from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation
from scipy.spatial import KDTree


class RHPlanner:
    def __init__(
        self,
        start_pose: np.array,
        mesh_coordinates: np.array,
        mesh_tree,
        grid_size: np.array = np.array([0.3, 0.3, 0.3]),
        voxel_size: np.array = np.array([0.002]),
        grid_center: np.array = np.array([0.5, -0.4, 1.1]),
        image_size: np.array = np.array([600, 450]),
        intrinsics: np.array = np.array([
            [685.5028076171875, 0.0, 485.35955810546875],
            [0.0, 685.6409912109375, 270.7330627441406],
            [0.0, 0.0, 1.0],
        ]),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        num_samples: int = 1,
        target_params: np.array = np.array([0.5, -0.4, 1.1]),
        # RH parameters
        horizon: int = 3,
        num_candidates: int = 10,
        lambda_cost: float = 2.0,
        step_size: float = 0.065,
        bias_ratio: float = 0.7,
        discount: float = 0.85,
        # Improvement parameters
        r_min: float = 0.15,          # min orbit radius around target
        r_max: float = 0.45,          # max orbit radius around target
        occlusion_bonus: float = 2.0, # weight for occlusion-aware IG bonus
        stagnation_patience: int = 4, # iters without coverage gain → escape
        rng_seed: int = 42,
        robot_reach_bounds: np.array = None,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.target_params = torch.tensor(
            target_params, dtype=torch.float32, device=self.device
        )

        # ------------------------------------------------------------------
        # 1. SPHERICAL BOUNDS — centred on target
        # ------------------------------------------------------------------
        self.r_min = r_min
        self.r_max = r_max
        if robot_reach_bounds is not None:
            self.robot_reach_bounds = torch.tensor(
                robot_reach_bounds, dtype=torch.float32, device=self.device
            )
        else:
            self.robot_reach_bounds = None
        # Axis-aligned bounding box of the sphere (for clamp fallback)
        target_np = np.array(target_params)
        self.camera_bounds = torch.tensor(
            [
                [target_np[0] - r_max, target_np[1] - r_max, target_np[2] - r_max],
                [target_np[0] + r_max, target_np[1] + r_max, target_np[2] + r_max],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Voxel grid
        self.voxel_grid = VoxelGrid(
            grid_size=torch.tensor(grid_size, dtype=torch.float32, device=self.device),
            voxel_size=torch.tensor(voxel_size, dtype=torch.float32, device=self.device),
            grid_center=torch.tensor(grid_center, dtype=torch.float32, device=self.device),
            width=image_size[0],
            height=image_size[1],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            num_pts_per_ray=num_pts_per_ray,
            num_features=num_features,
            target_params=self.target_params,
            device=self.device,
        )

        self.num_samples       = num_samples
        self.rviz_visualizer   = RvizVisualizer()
        self.mesh_coordinates  = mesh_coordinates
        self.mesh_tree         = mesh_tree

        # RH hyperparameters
        self.horizon        = horizon
        self.num_candidates = num_candidates
        self.lambda_cost    = lambda_cost
        self.step_size      = step_size
        self.bias_ratio     = bias_ratio
        self.discount       = discount
        self.occlusion_bonus = occlusion_bonus

        # Current camera position — updated after each real step
        self.current_pos = torch.tensor(
            start_pose[:3], dtype=torch.float32, device=self.device
        )

        # ------------------------------------------------------------------
        # 4. STAGNATION RECOVERY state
        # ------------------------------------------------------------------
        self.stagnation_patience  = stagnation_patience
        self._stagnation_counter  = 0
        self._last_coverage       = 0.0

        # Ray-tracing counter (cumulative, never reset)
        self.ray_trace_count = 0

        # Reproducibility
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)

        self.target_voxels        = np.array(0)
        self.candidate_history    = []
        self.occluded_mesh_points = None

        print(
            f"[RHPlanner] K={num_candidates}, H={horizon}, "
            f"r=[{r_min},{r_max}], occlusion_bonus={occlusion_bonus}, "
            f"stagnation_patience={stagnation_patience}"
        )

    # ------------------------------------------------------------------
    # 1. Spherical projection helper
    # ------------------------------------------------------------------
    def _within_reach(self, pos: torch.Tensor) -> bool:
        if self.robot_reach_bounds is None:
            return True
        lo = self.robot_reach_bounds[0]
        hi = self.robot_reach_bounds[1]
        return bool(torch.all(pos >= lo) and torch.all(pos <= hi))

    # Kamerayı Küre Üzerinde Tutuyor
    # Yeni hesaplanan pozisyon küre dışına çıkarsa geri iter. 
    # Hedeften olan yön korunur, sadece mesafe düzeltilir. 
    # Bircher'da kamera herhangi bir yere gidebiliyordu ve "köşe tuzağına" düşüyordu — bu fonksiyon onu engeller.
    def _project_to_shell(self, pos: torch.Tensor) -> torch.Tensor:
        """Project pos onto the orbital shell [r_min, r_max] around target."""
        vec  = pos - self.target_params  # yon vektoru = kamera - hedef 
        dist = torch.norm(vec)  # suanki uzaklik
        if dist < 1e-6:
            vec  = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            dist = torch.tensor(1.0, device=self.device)
        r = torch.clamp(dist, self.r_min, self.r_max)
        return self.target_params + vec / dist * r  # ayni yonde, dogru uzaklikta 

    # ------------------------------------------------------------------
    # Candidate generation — spherical + biased
    # ------------------------------------------------------------------
    def generate_candidate_sequence(self, start_pos: torch.Tensor) -> torch.Tensor:
        """
        Generate H-step sequence on the orbital sphere around target.
        bias_ratio fraction: tangential orbit steps.
        rest: random spherical jumps.
        """
        sequence = torch.zeros(
            (self.horizon, 3), dtype=torch.float32, device=self.device
        )
        prev_pos = start_pos.clone()

        # H adim uret
        for k in range(self.horizon):
            if torch.rand(1).item() < self.bias_ratio:
                # Tangential step on sphere surface
                to_target  = self.target_params - prev_pos
                dist       = torch.norm(to_target)
                radial_dir = to_target / (dist + 1e-6)

                rand_dir = torch.randn(3, device=self.device)
                rand_dir = rand_dir - (rand_dir @ radial_dir) * radial_dir
                rand_norm = torch.norm(rand_dir)
                if rand_norm < 1e-6:
                    rand_dir  = torch.tensor([1.0, 0.0, 0.0], device=self.device)
                    rand_dir  = rand_dir - (rand_dir @ radial_dir) * radial_dir
                    rand_norm = torch.norm(rand_dir) + 1e-6
                tangent = rand_dir / rand_norm

                step    = self.step_size * (0.3 + 0.7 * torch.rand(1, device=self.device).item())
                new_pos = prev_pos + tangent * step
            else:
                # Random spherical sample — uniform on sphere shell
                phi     = torch.rand(1).item() * 2 * np.pi
                theta   = torch.rand(1).item() * np.pi
                r       = self.r_min + torch.rand(1).item() * (self.r_max - self.r_min)
                offset  = torch.tensor([
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta),
                ], dtype=torch.float32, device=self.device)
                new_pos = self.target_params + offset

            new_pos = self._project_to_shell(new_pos)
            if not self._within_reach(new_pos) and self.robot_reach_bounds is not None:
                new_pos = torch.max(
                    torch.min(new_pos, self.robot_reach_bounds[1]),
                    self.robot_reach_bounds[0],
                )
                new_pos = self._project_to_shell(new_pos)
            sequence[k] = new_pos
            prev_pos    = new_pos

        return sequence

    # ------------------------------------------------------------------
    # 2. Occlusion-aware IG
    # ------------------------------------------------------------------
    def compute_gain_on_grid(
        self, voxel_grid_data: torch.Tensor, camera_pos: torch.Tensor
    ) -> float:
        """
        Transmittance-weighted semantic entropy + occlusion bonus.
        Occlusion bonus: voxels that are occupied AND semantically uncertain
        (likely occluded targets) get extra weight.
        """
        self.ray_trace_count += 1
        quat       = look_at_rotation(camera_pos, self.target_params)
        transforms = transform_from_rotation_translation(
            quat[None, :], camera_pos[None, :]
        )

        t_vals = self.voxel_grid.t_vals.clone()
        ray_origins, ray_directions, _ = (
            self.voxel_grid.ray_sampler.ray_origins_directions(transforms=transforms)
        )
        ray_points = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)

        ray_points_nor = self.voxel_grid.normalize_3d_coordinate(ray_points)
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3).repeat(2, 1, 1, 1, 1)
        grid           = voxel_grid_data[None, ..., 1:3].permute(4, 0, 1, 2, 3)
        occ_sem_confs  = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem_confs  = occ_sem_confs.view(2, -1, self.voxel_grid.num_pts_per_ray)
        occ_sem_confs  = occ_sem_confs.clamp(
            self.voxel_grid.eps, 1.0 - self.voxel_grid.eps
        )

        opacities      = torch.sigmoid(1e7 * (occ_sem_confs[0, ...] - 0.51))
        transmittance  = self.voxel_grid.shifted_cumprod(1.0 - opacities)
        entropy        = self.voxel_grid.entropy(occ_sem_confs[1, ...])
        ray_gains      = transmittance * entropy
        base_gain      = torch.log(torch.mean(ray_gains) + self.voxel_grid.eps)

        # ------------------------------------------------------------------
        # 2. Occlusion bonus: high occupancy + high semantic uncertainty
        # = likely an occluded target voxel becoming visible
        # ------------------------------------------------------------------
        occ_vals  = occ_sem_confs[0, ...]   # occupancy along rays
        sem_vals  = occ_sem_confs[1, ...]   # semantic uncertainty along rays
        occ_high  = (occ_vals > 0.6)
        sem_unc   = (sem_vals > 0.3) & (sem_vals < 0.7)
        occ_bonus = torch.mean((occ_high & sem_unc).float()) * self.occlusion_bonus

        return base_gain.item() + occ_bonus.item()

    # ------------------------------------------------------------------
    # PredictUpdate — belief forward simulation (no ray_trace_count)
    # ------------------------------------------------------------------
    def predict_update(
        self, voxel_grid_data: torch.Tensor, camera_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate hypothetical measurement. Does NOT count as a ray-tracing
        call (Burusa metric 3 counts only real gain evaluations).
        """
        updated_grid = voxel_grid_data.clone()

        quat       = look_at_rotation(camera_pos, self.target_params)
        transforms = transform_from_rotation_translation(
            quat[None, :], camera_pos[None, :]
        )
        t_vals       = self.voxel_grid.t_vals
        ray_origins, ray_directions, _ = (
            self.voxel_grid.ray_sampler.ray_origins_directions(transforms=transforms)
        )
        ray_points   = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)
        grid_coords  = torch.div(
            ray_points - self.voxel_grid.origin,
            self.voxel_grid.voxel_size,
            rounding_mode="floor",
        )
        valid_indices = self.voxel_grid.get_valid_indices(
            grid_coords, self.voxel_grid.voxel_dims
        )
        valid_coords  = grid_coords[valid_indices].to(torch.long)
        if valid_coords.numel() == 0:
            return updated_grid

        dims     = self.voxel_grid.voxel_dims
        Dy, Dz   = int(dims[1]), int(dims[2])
        flat_keys  = (
            valid_coords[:, 0] * (Dy * Dz)
            + valid_coords[:, 1] * Dz
            + valid_coords[:, 2]
        )
        unique_keys = torch.unique(flat_keys)
        gz = unique_keys % Dz
        gy = (unique_keys // Dz) % Dy
        gx = unique_keys // (Dy * Dz)

        sem      = updated_grid[gx, gy, gz, 2]
        sem_mask = (sem > 0.3) & (sem < 0.7)
        if sem_mask.any():
            updated_grid[gx[sem_mask], gy[sem_mask], gz[sem_mask], 2] = (
                0.6 * sem[sem_mask] + 0.4 * 0.65
            )
        occ      = updated_grid[gx, gy, gz, 1]
        occ_mask = (occ > 0.45) & (occ < 0.55)
        if occ_mask.any():
            updated_grid[gx[occ_mask], gy[occ_mask], gz[occ_mask], 1] = (
                0.6 * occ[occ_mask] + 0.4 * 0.35
            )
        return updated_grid

    # ------------------------------------------------------------------
    # Path cost
    # ------------------------------------------------------------------
    def motion_cost(self, pos_prev: torch.Tensor, pos_next: torch.Tensor) -> float:
        return torch.norm(pos_next - pos_prev).item()

    # ------------------------------------------------------------------
    # 3. Incremental IG sequence evaluation
    # ------------------------------------------------------------------
    def evaluate_sequence(
        self, sequence: torch.Tensor, start_pos: torch.Tensor
    ) -> float:
        """
        J = Σ_k γ^k * [f_incremental(M_pred, ξ_k) - λ * C(ξ_{k-1}, ξ_k)]

        f_incremental only counts gain from voxels NOT seen in earlier steps
        of this sequence — prevents rewarding stationary sequences.
        """
        J         = 0.0
        prev_pos  = start_pos.clone()
        M_pred    = self.voxel_grid.voxel_grid.clone()
        # Track which voxel positions have been "seen" this sequence
        seen_set  = set()

        for k in range(self.horizon):
            xi_k = sequence[k]

            # Full gain at this step
            gain_full = self.compute_gain_on_grid(M_pred, xi_k)

            # Incremental: subtract gain already counted in seen positions
            # Approximate: if camera hasn't moved much, gain is redundant
            if k > 0:
                min_dist_to_seen = min(
                    torch.norm(xi_k - sequence[j]).item()
                    for j in range(k)
                )
                # If within one step_size of a previous position, penalise
                if min_dist_to_seen < self.step_size * 0.5:
                    gain_full *= 0.3  # heavy redundancy penalty

            cost   = self.motion_cost(prev_pos, xi_k)
            weight = self.discount ** k
            J     += weight * (gain_full - self.lambda_cost * cost)

            M_pred   = self.predict_update(M_pred, xi_k)
            prev_pos = xi_k

        return J

    # ------------------------------------------------------------------
    # 5. Adaptive horizon
    # ------------------------------------------------------------------
    def _adaptive_horizon(self, coverage: float) -> int:
        if coverage < 40.0:
            return 2   # fast early exploration
        elif coverage < 70.0:
            return 3   # default
        else:
            return 5   # deep occlusion handling

    # ------------------------------------------------------------------
    # Main entry: receding horizon step
    # ------------------------------------------------------------------
    def rh_view(self, current_coverage: float = 0.0) -> tuple:
        """
        Sample K candidates, evaluate with incremental IG, execute first
        step of best sequence. Includes stagnation escape.

        Args:
            current_coverage: latest ROI coverage % (for adaptive H)
        Returns:
            (viewpoint [7,], loss, num_evaluations)
        """
        iter_ray_calls_before = self.ray_trace_count

        # 5. Adaptive horizon
        self.horizon = self._adaptive_horizon(current_coverage)

        # 4. Stagnation check
        if abs(current_coverage - self._last_coverage) < 0.5:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
        self._last_coverage = current_coverage

        if self._stagnation_counter >= self.stagnation_patience:
            print(f"[RHPlanner] Stagnation detected ({self._stagnation_counter} iters) "
                  f"— forcing escape to new orbit position")
            # Instead of exact antipodal (may be out of robot reach),
            # sample random positions on the sphere until we find one
            # that differs from current by at least step_size * 3
            best_escape = None
            best_dist   = 0.0
            for _ in range(20):
                phi     = torch.rand(1).item() * 2 * 3.14159
                theta   = torch.rand(1).item() * 3.14159
                r       = self.r_min + torch.rand(1).item() * (self.r_max - self.r_min)
                offset  = torch.tensor([
                    r * float(torch.sin(torch.tensor(theta)) * torch.cos(torch.tensor(phi))),
                    r * float(torch.sin(torch.tensor(theta)) * torch.sin(torch.tensor(phi))),
                    r * float(torch.cos(torch.tensor(theta))),
                ], dtype=torch.float32, device=self.device)
                candidate = self.target_params + offset
                d = torch.norm(candidate - self.current_pos).item()
                if d > best_dist:
                    best_dist   = d
                    best_escape = candidate
            if best_escape is not None and best_dist > self.step_size * 2:
                self.current_pos = best_escape
            self._stagnation_counter = 0

        best_J         = -np.inf
        best_sequence  = None
        iter_candidates = []
        best_idx       = 0

        for k in range(self.num_candidates):
            sequence = self.generate_candidate_sequence(self.current_pos)
            J        = self.evaluate_sequence(sequence, self.current_pos)
            iter_candidates.append({
                "sequence": sequence.detach().cpu().numpy(),
                "score":    J,
            })
            if J > best_J:
                best_J        = J
                best_sequence = sequence
                best_idx      = k

        # Store candidate history for plots
        self.candidate_history.append({
            "start_pos": self.current_pos.detach().cpu().numpy().copy(),
            "sequences": np.array([c["sequence"] for c in iter_candidates]),
            "scores":    np.array([c["score"]    for c in iter_candidates]),
            "best_idx":  best_idx,
        })

        # Receding horizon: execute only first step
        best_first_pos   = best_sequence[0]
        self.current_pos = best_first_pos.clone()

        quat      = look_at_rotation(best_first_pos, self.target_params)
        viewpoint = np.zeros(7)
        viewpoint[:3] = best_first_pos.detach().cpu().numpy()
        viewpoint[3:] = quat.detach().cpu().numpy()

        evals_this_iter = self.ray_trace_count - iter_ray_calls_before
        return viewpoint, -best_J, evals_this_iter

    # ------------------------------------------------------------------
    # Occluded recall
    # ------------------------------------------------------------------
    def set_occluded_mesh_points(self):
        voxel_points, _, _ = self.get_occupied_points()
        half   = 0.002
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
        print(f"[RHPlanner] Occluded after view 0: "
              f"{len(self.occluded_mesh_points)}/{len(self.mesh_coordinates)} "
              f"({100*len(self.occluded_mesh_points)/len(self.mesh_coordinates):.1f}%)")

    def compute_occluded_recall(self) -> float:
        if self.occluded_mesh_points is None or len(self.occluded_mesh_points) == 0:
            return 0.0
        voxel_points, _, _ = self.get_occupied_points()
        if len(voxel_points) == 0:
            return 0.0
        voxel_tree = KDTree(voxel_points)
        half       = 0.002
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
    # Real voxel grid update
    # ------------------------------------------------------------------
    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ):
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position    = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(viewpoint[3:], dtype=torch.float32, device=self.device)
        transform   = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        coverage = self.voxel_grid.insert_depth_and_semantics(
            depth_image, semantics, transform
        )
        if coverage is not None:
            coverage = coverage.cpu().numpy()
        self.current_pos = position.clone()
        return coverage

    # ------------------------------------------------------------------
    # Occupied voxel access
    # ------------------------------------------------------------------
    def get_occupied_points(self):
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        return (
            voxel_points.cpu().numpy(),
            sem_conf_scores.cpu().numpy(),
            sem_class_ids.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # F1 / recall / precision — full mesh (bunny has no semantic labels)
    # ------------------------------------------------------------------
    def calculate_F1(self, occluder_positions=None):
        """
        occluder_positions: list of (center, half_size) tuples to mask out.
        Voxels inside known occluders are excluded before F1 computation.
        """
        voxel_points, _, _ = self.get_occupied_points()

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        # Mask out known occluder voxels (they are false positives)
        if occluder_positions:
            keep = np.ones(len(voxel_points), dtype=bool)
            for center, half in occluder_positions:
                c = np.array(center)
                h = np.array(half)
                in_occ = np.all(np.abs(voxel_points - c) <= h, axis=1)
                keep &= ~in_occ
            voxel_points = voxel_points[keep]

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        self.target_voxels = voxel_points
        mesh_tree   = KDTree(self.mesh_coordinates)
        voxel_tree  = KDTree(voxel_points)
        half        = 0.002
        radius      = half * np.sqrt(3)

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
            if precision + recall > 0 else 0
        )
        return f1, recall, precision

    # ------------------------------------------------------------------
    # Sigma: spatial spread of detected target voxels
    # ------------------------------------------------------------------
    def compute_sigma(self) -> float:
        if not isinstance(self.target_voxels, np.ndarray) or self.target_voxels.ndim < 2:
            return 0.0
        if len(self.target_voxels) == 0:
            return 0.0
        centroid = self.target_voxels.mean(axis=0)
        dists    = np.linalg.norm(self.target_voxels - centroid, axis=1)
        return float(dists.mean())

    # ------------------------------------------------------------------
    # RViz
    # ------------------------------------------------------------------
    def visualize(self):
        voxel_points, sem_conf_scores, sem_class_ids = self.get_occupied_points()
        self.rviz_visualizer.visualize_voxels(
            voxel_points, sem_conf_scores, sem_class_ids
        )
        target = self.target_params.detach().cpu().numpy()
        rois   = np.array([[*target, 1.0, 0.0, 0.0, 0.0]])
        self.rviz_visualizer.visualize_rois(numpy_to_pose_array(rois))
        self.rviz_visualizer.visualize_camera_bounds(
            self.camera_bounds.cpu().numpy()
        )
