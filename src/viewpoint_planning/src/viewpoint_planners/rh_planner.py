import os
import torch
import torch.nn.functional as F
import numpy as np

from scene_representation.voxel_grid import VoxelGrid
from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation, quaternion_to_matrix
from scipy.spatial import KDTree
try:
    from fair_comparison_config import ROI_HALF
except ModuleNotFoundError:
    from viewpoint_planners.fair_comparison_config import ROI_HALF


class RHPlanner:
    def __init__(
        self,
        start_pose: np.array,
        mesh_coordinates: np.array,
        mesh_tree,
        grid_size: np.array = np.array([0.3, 0.6, 0.3]),
        voxel_size: np.array = np.array([0.003]),
        grid_center: np.array = np.array([0.5, -0.25, 1.1]),
        image_size: np.array = np.array([600, 450]),
        intrinsics: np.array = np.array([
            [685.5028076171875, 0.0, 485.35955810546875],
            [0.0, 685.6409912109375, 270.7330627441406],
            [0.0, 0.0, 1.0],
        ]),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        num_samples: int = 1,
        target_params: np.array = np.array([0.5, -0.25, 1.1]),
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
        occlusion_bonus: float = 2.0, # weight for occlusion-aware IG bonus.
                                      # DEFAULT 0.0 = OFF: GradientNBV's utility
                                      # has no such term, so it is disabled for
                                      # the fair baseline comparison. Set >0
                                      # (e.g. 2.0) to re-enable as an ablation.
        stagnation_patience: int = 4,   # iters without coverage gain → then escape
        stagnation_threshold: float = 1.5,  # min % gain to count as non-stagnant
        rng_seed: int = 42,
        robot_reach_bounds: np.array = None,
        # DEFAULT False = OFF: the spherical orbital shell is an RH-specific
        # constraint absent from Burusa's GradientNBV. It is disabled by default
        # so both planners search the SAME box-constrained camera space. Set
        # True to re-enable the shell as an ablation (reproduces older results).
        use_spherical_bounds: bool = False,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.target_params = torch.tensor(
            target_params, dtype=torch.float32, device=self.device
        )


        # ---------SPHERICAL BOUNDS - centred on target-----------
        self.r_min = r_min
        self.r_max = r_max
        self.use_spherical_bounds = use_spherical_bounds
        if robot_reach_bounds is not None:
            self.robot_reach_bounds = torch.tensor(
                robot_reach_bounds, dtype=torch.float32, device=self.device
            )
        else:
            self.robot_reach_bounds = None
        # Axis-aligned camera bounds.
        # MATCHED EXACTLY to GradientNBVPlanner's camera box so both planners
        # search the same physical camera workspace (fair comparison). Burusa's
        # planner uses start_pose +/- CAMERA_BOUNDS_HALFWIDTHS; we mirror that here
        # instead of the old target +/- r_max cube. The shell (if re-enabled via
        # use_spherical_bounds) still uses r_min/r_max independently.
        start_np = np.asarray(start_pose[:3], dtype=np.float32)
        from viewpoint_planners.fair_comparison_config import CAMERA_BOUNDS_HALFWIDTHS
        bounds_halfwidths = CAMERA_BOUNDS_HALFWIDTHS
        self.camera_bounds = torch.tensor(
            [
                (start_np - bounds_halfwidths).tolist(),
                (start_np + bounds_halfwidths).tolist(),
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

        # Current camera position: updated after each real step
        self.current_pos = torch.tensor(
            start_pose[:3], dtype=torch.float32, device=self.device
        )


        self.stagnation_patience   = stagnation_patience
        self.stagnation_threshold  = stagnation_threshold
        self._stagnation_counter   = 0
        self._last_coverage        = 0.0

        # Ray-tracing counter (cumulative, never reset)
        self.ray_trace_count = 0

        # Reproducibility
        np.random.seed(rng_seed)
        torch.manual_seed(rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rng_seed)

        self.target_voxels        = np.array(0)
        self.all_target_voxels    = np.zeros((0, 3))  # full recon for plotting
        self.candidate_history    = []
        self.occluded_mesh_points = None

        # Explicit TP/FP/FN from the most recent F1 computation (Burusa-style).
        self.last_tp = 0
        self.last_fp = 0
        self.last_fn = 0

        print(
            f"[RHPlanner] K={num_candidates}, H={horizon}, "
            f"r=[{r_min},{r_max}], occlusion_bonus={occlusion_bonus}, "
            f"stagnation_patience={stagnation_patience}, "
            f"stagnation_threshold={stagnation_threshold}%"
        )


    # SPHERICAL BOUNDS — camera orbits target on a sphere (r_min, r_max).
    # No more corner-trapping. Positions outside the shell are re-projected back onto the sphere surface.
    def _within_reach(self, pos: torch.Tensor) -> bool:
        if self.robot_reach_bounds is None:
            return True
        lo = self.robot_reach_bounds[0]
        hi = self.robot_reach_bounds[1]
        return bool(torch.all(pos >= lo) and torch.all(pos <= hi))

    # Kamerayı Küre Üzerinde Tutuyor
    # Yeni hesaplanan pozisyon küre dışına çıkarsa geri iter. 
    # Hedeften olan yön korunur, sadece mesafe düzeltilir. 
    # Bircher'da kamera herhangi bir yere gidebiliyordu ve "köşe tuzağına" düşüyordu, bu fonksiyon onu engelliyor.
    def _project_to_shell(self, pos: torch.Tensor) -> torch.Tensor:
        """Project pos onto the orbital shell [r_min, r_max] around target.

        If use_spherical_bounds is False, the shell is disabled and pos is
        returned unchanged (Burusa-style: only the reach box constrains pos).
        """
        if not self.use_spherical_bounds:
            return pos
        vec  = pos - self.target_params  # yon vektoru = kamera - hedef 
        dist = torch.norm(vec)  # suanki uzaklik
        if dist < 1e-6:
            vec  = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            dist = torch.tensor(1.0, device=self.device)
        r = torch.clamp(dist, self.r_min, self.r_max)
        return self.target_params + vec / dist * r  # ayni yonde, dogru uzaklikta 

    # ------------------------------------------------------------------
    # Candidate generation
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
                if self.use_spherical_bounds:
                    # Random spherical sample on the orbital shell.
                    phi     = torch.rand(1).item() * 2 * np.pi
                    theta   = torch.rand(1).item() * np.pi
                    r       = self.r_min + torch.rand(1).item() * (self.r_max - self.r_min)
                    offset  = torch.tensor([
                        r * np.sin(theta) * np.cos(phi),
                        r * np.sin(theta) * np.sin(phi),
                        r * np.cos(theta),
                    ], dtype=torch.float32, device=self.device)
                    new_pos = self.target_params + offset
                else:
                    # Burusa-style: uniform sample inside the reach box.
                    if self.robot_reach_bounds is not None:
                        lo, hi = self.robot_reach_bounds[0], self.robot_reach_bounds[1]
                    else:
                        lo, hi = self.camera_bounds[0], self.camera_bounds[1]
                    new_pos = lo + torch.rand(3, device=self.device) * (hi - lo)

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

  
    # Occlusion-aware IG
    @torch.no_grad()
    def compute_gain_on_grid(
        self, voxel_grid_data: torch.Tensor, camera_pos: torch.Tensor
    ) -> float:
        """
        Transmittance-weighted semantic entropy + occlusion bonus.
        Occlusion bonus: voxels that are occupied & semantically uncertain
        (likely occluded targets) get extra weight.

        Wrapped in torch.no_grad(): this function only returns a scalar via
        .item() and never back-propagates, so building the autograd graph was
        pure overhead. Disabling it leaves every numeric result identical while
        cutting memory and runtime (important on the limited-GPU laptop).
        """
        self.ray_trace_count += 1
        quat       = look_at_rotation(camera_pos, self.target_params)
        transforms = transform_from_rotation_translation(
            quat[None, :], camera_pos[None, :]
        )

        # t_vals is read-only here, so the previous .clone() was unnecessary.
        t_vals = self.voxel_grid.t_vals
        ray_origins, ray_directions, _ = (
            self.voxel_grid.ray_sampler.ray_origins_directions(transforms=transforms)
        )
        ray_points = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)

        ray_points_nor = self.voxel_grid.normalize_3d_coordinate(ray_points)
        del ray_points  # free ~395 MB before grid_sample to avoid OOM
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3)
        # Use (1,2,Dx,Dy,Dz) grid so grid_sample handles both channels in one
        # batch-1 call — eliminates the .repeat(2,...) that duplicated the
        # 395 MB query tensor and caused OOM on the 16 GB laptop.
        grid          = voxel_grid_data[None, ..., 1:3].permute(0, 4, 1, 2, 3)
        occ_sem_confs = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem_confs = occ_sem_confs.view(2, -1, self.voxel_grid.num_pts_per_ray)
        occ_sem_confs  = occ_sem_confs.clamp(
            self.voxel_grid.eps, 1.0 - self.voxel_grid.eps
        )

        opacities      = torch.sigmoid(1e7 * (occ_sem_confs[0, ...] - 0.51))
        transmittance  = self.voxel_grid.shifted_cumprod(1.0 - opacities)
        entropy        = self.voxel_grid.entropy(occ_sem_confs[1, ...])
        ray_gains      = transmittance * entropy
        base_gain      = torch.log(torch.mean(ray_gains) + self.voxel_grid.eps)

       
        # Occlusion bonus: high occupancy + high semantic uncertainty
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
    @torch.no_grad()
    def predict_update(
        self, voxel_grid_data: torch.Tensor, camera_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate hypothetical measurement. Does NOT count as a ray-tracing
        call (Burusa metric 3 counts only real gain evaluations).

        Wrapped in torch.no_grad(): the predicted grid is only consumed by
        compute_gain_on_grid (also no-grad), never differentiated, so the
        autograd graph was unused overhead. Results are unchanged.
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
        del ray_points, grid_coords, valid_indices  # free ~820 MB before unique/indexing
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
    # Receding horizon step
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

        # 5. Adaptive horizon — DISABLED for controlled ablation.
        # H must stay fixed at the value set in __init__ so that the
        # ablation study over H in {1,2,3,4} is a controlled variable.
        # To re-enable adaptive H as a separate experiment, uncomment:
        # if self.use_adaptive_horizon:
        #     self.horizon = self._adaptive_horizon(current_coverage)

        # 4. Stagnation check
        if abs(current_coverage - self._last_coverage) < self.stagnation_threshold:
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
                if self.use_spherical_bounds:
                    phi     = torch.rand(1).item() * 2 * 3.14159
                    theta   = torch.rand(1).item() * 3.14159
                    r       = self.r_min + torch.rand(1).item() * (self.r_max - self.r_min)
                    offset  = torch.tensor([
                        r * float(torch.sin(torch.tensor(theta)) * torch.cos(torch.tensor(phi))),
                        r * float(torch.sin(torch.tensor(theta)) * torch.sin(torch.tensor(phi))),
                        r * float(torch.cos(torch.tensor(theta))),
                    ], dtype=torch.float32, device=self.device)
                    candidate = self.target_params + offset
                else:
                    # Burusa-style: uniform escape inside the reach box.
                    if self.robot_reach_bounds is not None:
                        lo, hi = self.robot_reach_bounds[0], self.robot_reach_bounds[1]
                    else:
                        lo, hi = self.camera_bounds[0], self.camera_bounds[1]
                    candidate = lo + torch.rand(3, device=self.device) * (hi - lo)
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


    # Occluded recall
    def set_occluded_mesh_points(self):
        voxel_points, _, _ = self.get_occupied_points()
        vsize = float(np.asarray(self.voxel_grid.voxel_size.detach().cpu().numpy()).reshape(-1)[0])
        half   = vsize * 4.0
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
        vsize      = float(np.asarray(self.voxel_grid.voxel_size.detach().cpu().numpy()).reshape(-1)[0])
        half       = vsize * 4.0
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


    # D455: camera_link (Gazebo sensor) is 59mm offset from camera_color_frame (MoveIt tip).
    # camera_link = camera_color_frame + R_cws @ [0, +0.059, 0]
    _D455_COLOR_TO_DEPTH = torch.tensor([0.0, 0.059, 0.0])

    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ):
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position    = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(viewpoint[3:], dtype=torch.float32, device=self.device)
        R_cws = quaternion_to_matrix(orientation[None, :])[0]
        position = position + R_cws @ self._D455_COLOR_TO_DEPTH.to(self.device)
        transform   = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        coverage = self.voxel_grid.insert_depth_and_semantics(
            depth_image, semantics, transform
        )
        if coverage is not None and hasattr(coverage, "cpu"):
            coverage = float(coverage.cpu().numpy())
        self.current_pos = position.clone()
        return coverage


    # Occupied voxel access
    def get_occupied_points(self):
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        return (
            voxel_points.cpu().numpy(),
            sem_conf_scores.cpu().numpy(),
            sem_class_ids.cpu().numpy(),
        )


    # F1 / recall / precision — Burusa Table II aligned
    def calculate_F1(self, occluder_positions=None, match_threshold=None,
                     diagnose=False):
        """
        F1 of 3D node reconstruction, aligned with Burusa et al. (ICRA 2024):
          - Only voxels of the target class (fruit node, class id 0) are scored.
          - Both reconstructed voxels and ground-truth mesh are clipped to the
            6cm ROI cube centred on the target before matching.
          - A reconstructed point is a true positive (TP) when within
            `match_threshold` of a ground-truth mesh point.

        match_threshold: TP distance threshold in metres. MUST be >= the voxel
            resolution, otherwise grid-snapped voxel centres can never land
            within threshold of the continuous mesh surface and TP is always 0.
            Burusa uses 2mm because their voxel grid is 2mm; here the grid is
            `self.voxel_grid.voxel_size`, so we default to that value.
        diagnose: if True, print the voxel->mesh nearest-distance distribution.
            Use this once to confirm whether F1=0 is a threshold issue or a
            real coordinate/calibration offset between voxels and mesh.

        Sets self.last_tp / self.last_fp / self.last_fn for explicit reporting.
        Returns (f1, recall, precision) for backward compatibility.
        """
        voxel_points, _, sem_class = self.get_occupied_points()

        self.last_tp = 0
        self.last_fp = 0
        self.last_fn = 0

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        # 1) Keep only target-class voxels (Burusa: class 0 = fruit node).
        #    sem_class may be empty/scalar in degenerate early frames — guard it.
        sem_class = np.asarray(sem_class)
        n_class0 = int(np.sum(sem_class == 0)) if sem_class.shape[0] == voxel_points.shape[0] else -1
        print(f"  [F1 DIAG] occupied={len(voxel_points)} class0={n_class0} class-1={int(np.sum(sem_class==-1)) if n_class0>=0 else '?'}")
        if sem_class.shape[0] == voxel_points.shape[0]:
            target_mask = (sem_class == 0)
            voxel_points = voxel_points[target_mask]

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        # 2) Mask out known occluder voxels (false positives from occluders).
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

        # Keep the full set of reconstructed target-class voxels (before the ROI
        # clip) for visualisation. F1 below is scored only inside the ROI, but
        # the reconstruction plot should show everything the camera actually
        # recovered of the object surface, not just the ROI slice.
        self.all_target_voxels = voxel_points.copy()

        # 3) Clip both voxels and mesh to the ROI cube around the target.
        #    IMPORTANT: this must match the coverage ROI used in voxel_grid.py
        #    (set_target_roi uses +/-25 voxels = +/-75mm at 3mm voxels). Using a
        #    smaller cube here (the old 30mm) made F1 score only a thin central
        #    slice while coverage counted the whole 150mm region, which is why
        #    the reconstruction plot looked like a thin strip and F1/coverage
        #    disagreed. Keep both ROIs identical. Overridable via ROI_HALF env.
        target   = self.target_params.detach().cpu().numpy()
        roi_half = float(os.environ.get("ROI_HALF", ROI_HALF))
        v_in_roi = np.all(np.abs(voxel_points - target) <= roi_half, axis=1)
        voxel_points = voxel_points[v_in_roi]

        m_in_roi = np.all(np.abs(self.mesh_coordinates - target) <= roi_half, axis=1)
        roi_mesh = self.mesh_coordinates[m_in_roi]

        if len(voxel_points) == 0 or len(roi_mesh) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0, 0, 0

        self.target_voxels = voxel_points
        mesh_tree  = KDTree(roi_mesh)
        voxel_tree = KDTree(voxel_points)

        # Matching threshold for TP. The voxel grid is 3mm but reconstructed
        # voxel centres sit ~9-13mm from the continuous mesh surface (confirmed
        # for BOTH RH-NBV and Burusa's own GradientNBV in this setup), because
        # occupancy-thresholded voxels form a shell slightly in front of the
        # surface and the depth sensor is noisy. A 2-3mm threshold can therefore
        # never produce a TP. We default to a physically-honest value tied to
        # the voxel resolution plus observed surface offset: ceil to ~one voxel
        # diagonal of slack on top of the grid size. This is reported openly and
        # overridable via F1_THRESH (in metres) for sensitivity analysis.
        if match_threshold is None:
            vs = self.voxel_grid.voxel_size
            if hasattr(vs, "detach"):
                vs = vs.detach().cpu().numpy()
            vsize = float(np.asarray(vs).reshape(-1)[0])
            # grid resolution + surface-shell slack (default 4x voxel ~ 12mm).
            default_thr = float(os.environ.get("F1_THRESH", vsize * 4.0))
            match_threshold = default_thr
        half   = match_threshold
        radius = half * np.sqrt(3)

        # --- Optional diagnostic: voxel->mesh nearest-distance distribution ---
        # This reveals whether voxels actually lie ON the mesh surface (small
        # distances -> threshold problem) or are offset in space (large
        # distances -> coordinate/calibration mismatch).
        if diagnose:
            d_vox2mesh, nn_idx = mesh_tree.query(voxel_points)
            pct = np.percentile(d_vox2mesh, [0, 25, 50, 75, 100]) * 1000
            print("  [F1 DIAG] voxel->mesh nearest dist (mm): "
                  f"min={pct[0]:.1f} q25={pct[1]:.1f} med={pct[2]:.1f} "
                  f"q75={pct[3]:.1f} max={pct[4]:.1f}")
            print(f"  [F1 DIAG] threshold={half*1000:.1f}mm | "
                  f"voxels_in_ROI={len(voxel_points)} mesh_in_ROI={len(roi_mesh)} | "
                  f"frac voxels within thr="
                  f"{float(np.mean(d_vox2mesh <= half))*100:.1f}%")
            # Axis-resolved offset: mean signed (voxel - nearest_mesh) per axis.
            # A large consistent value on one axis points to a coordinate /
            # transform mismatch on that axis rather than random noise.
            nearest = roi_mesh[nn_idx]
            signed  = (voxel_points - nearest) * 1000  # mm
            print(f"  [F1 DIAG] mean signed offset (mm): "
                  f"x={signed[:,0].mean():+.1f} y={signed[:,1].mean():+.1f} "
                  f"z={signed[:,2].mean():+.1f}")
            print(f"  [F1 DIAG] voxel centroid: "
                  f"({voxel_points[:,0].mean():.3f}, {voxel_points[:,1].mean():.3f}, "
                  f"{voxel_points[:,2].mean():.3f}) | "
                  f"roi-mesh centroid: ({roi_mesh[:,0].mean():.3f}, "
                  f"{roi_mesh[:,1].mean():.3f}, {roi_mesh[:,2].mean():.3f})")
            # Bounding boxes reveal whether voxels are a sub-region of the mesh
            # (boxes overlap -> partial observation, alignment OK) or sit in a
            # different place (boxes disjoint -> coordinate mismatch).
            vlo, vhi = voxel_points.min(0), voxel_points.max(0)
            mlo, mhi = roi_mesh.min(0), roi_mesh.max(0)
            print(f"  [F1 DIAG] voxel  bbox: "
                  f"x[{vlo[0]:.3f},{vhi[0]:.3f}] y[{vlo[1]:.3f},{vhi[1]:.3f}] "
                  f"z[{vlo[2]:.3f},{vhi[2]:.3f}]")
            print(f"  [F1 DIAG] mesh   bbox: "
                  f"x[{mlo[0]:.3f},{mhi[0]:.3f}] y[{mlo[1]:.3f},{mhi[1]:.3f}] "
                  f"z[{mlo[2]:.3f},{mhi[2]:.3f}]")

        # True positives: reconstructed voxels matching a GT mesh point.
        nr_correct = 0
        for voxel in voxel_points:
            for idx in mesh_tree.query_ball_point(voxel, r=radius):
                coord = roi_mesh[idx]
                if all(abs(voxel[d] - coord[d]) <= half for d in range(3)):
                    nr_correct += 1
                    break

        # Recalled: GT mesh points covered by some reconstructed voxel.
        nr_recalled = 0
        for coord in roi_mesh:
            for idx in voxel_tree.query_ball_point(coord, r=radius):
                voxel = voxel_points[idx]
                if all(abs(voxel[d] - coord[d]) <= half for d in range(3)):
                    nr_recalled += 1
                    break

        # Explicit TP / FP / FN (supervisor request).
        self.last_tp = nr_correct
        self.last_fp = len(voxel_points) - nr_correct
        self.last_fn = len(roi_mesh) - nr_recalled

        precision = nr_correct  / len(voxel_points)
        recall    = nr_recalled / len(roi_mesh)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0
        )
        return f1, recall, precision


    # Sigma: spatial spread of detected target voxels
    def compute_sigma(self) -> float:
        if not isinstance(self.target_voxels, np.ndarray) or self.target_voxels.ndim < 2:
            return 0.0
        if len(self.target_voxels) == 0:
            return 0.0
        centroid = self.target_voxels.mean(axis=0)
        dists    = np.linalg.norm(self.target_voxels - centroid, axis=1)
        return float(dists.mean())


    # --------------RViz----------------
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
