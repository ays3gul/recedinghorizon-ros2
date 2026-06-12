import torch
import torch.nn as nn
import numpy as np
import time

from scene_representation.voxel_grid import VoxelGrid
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation

from viewpoint_planners.planner_eval_mixin import PlannerEvalMixin, init_eval_state
from viewpoint_planners.fair_comparison_config import (
    GRID_SIZE as FC_GRID_SIZE,
    VOXEL_SIZE as FC_VOXEL_SIZE,
)


class _NoOpVisualizer:
    """ROS 2 stub — RViz visualization calls are silently ignored."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class PsoPlanner(PlannerEvalMixin):
    """
    Particle Swarm Optimization viewpoint planner.
    ROS 2 Jazzy compatible — RvizVisualizer replaced with no-op stub.
    PSO strategy (c1/c2/w/bc, bouncing bounds, pbest/gbest) is UNCHANGED.
    """

    def __init__(
        self,
        start_pose: np.array,
        mesh_coordinates: np.array,
        mesh_tree,
        grid_size: np.array = FC_GRID_SIZE,
        voxel_size: np.array = FC_VOXEL_SIZE,
        grid_center: np.array = np.array([0.5, -0.4, 1.1]),
        image_size: np.array = np.array([600, 450]),
        intrinsics: np.array = np.array(
            [
                [685.5028076171875, 0.0, 485.35955810546875],
                [0.0, 685.6409912109375, 270.7330627441406],
                [0.0, 0.0, 1.0],
            ],
        ),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        num_samples: int = 1,
        target_params: np.array = np.array([0.5, -0.4, 1.1]),
        c1: float = 1.25,
        c2: float = 0.5,
        w: float = 1.5,
        bc: float = 0.75,
        n_particles: int = 4,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.optimization_params(start_pose, target_params)
        self.voxel_grid = VoxelGrid(
            grid_size=grid_size,
            voxel_size=voxel_size,
            grid_center=grid_center,
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
        self.num_samples = num_samples
        self.rviz_visualizer = _NoOpVisualizer()  # ROS 2: no-op stub

        self.mesh_coordinates = mesh_coordinates
        self.mesh_tree = mesh_tree

        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.bc = bc
        self.n_particles = n_particles
        np.random.seed(int(time.time()))

        random_values = np.random.rand(self.n_particles, 3)
        random_values[:, 0] = random_values[:, 0] * 0.4 + start_pose[0] - 0.2
        random_values[:, 1] = random_values[:, 1] * 0.2 + start_pose[1] - 0.1
        random_values[:, 2] = random_values[:, 2] * 0.3 + start_pose[2] - 0.15

        self.X = torch.tensor(
            random_values, dtype=torch.float32, device=self.device,
        )

        velocities = np.random.rand(self.n_particles, 3)
        for i in range(self.n_particles):
            if self.X[i][0] < self.target_params[0]:
                velocities[i][0] = velocities[i][0] * 0.12
            else:
                velocities[i][0] = velocities[i][0] * -0.12
            if self.X[i][1] < self.target_params[1]:
                velocities[i][1] = velocities[i][1] * 0.06
            else:
                velocities[i][1] = velocities[i][1] * -0.06
            if self.X[i][2] < self.target_params[2]:
                velocities[i][2] = velocities[i][2] * 0.09
            else:
                velocities[i][2] = velocities[i][2] * -0.09

        self.V = torch.tensor(velocities, dtype=torch.float32, device=self.device)
        self.init = True
        self.pbest = self.X
        self.pbest_obj = np.zeros(self.n_particles)

        self.recall = 3
        self.obj_history = np.array(
            [[[np.inf, []] for _ in range(self.recall)] for _ in range(n_particles)],
            dtype=object
        )

        i = 0
        for x in self.X:
            self.pbest_obj[i] = self.voxel_grid.compute_gain(x, self.target_params)[0]
            i += 1

        for i in range(len(self.obj_history)):
            self._push_obj_history(i, self.pbest_obj[i], self.X[i].detach().cpu().numpy())

        self.pbest = torch.tensor(
            np.array([min(particle, key=lambda x: x[0])[1] for particle in self.obj_history]),
            dtype=torch.float32, device=self.device,
        )
        self.pbest_obj = np.array(
            [min(particle, key=lambda x: x[0])[0] for particle in self.obj_history]
        )
        self.gbest = self.pbest[self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()
        self.particle_trajectories = [self.X.detach().cpu().numpy()]

        init_eval_state(self)

    def optimization_params(self, start_pose: np.array, target_params: np.array) -> None:
        self.target_params = torch.tensor(
            target_params, dtype=torch.float32, device=self.device,
        )
        self.camera_bounds = torch.tensor(
            [
                [start_pose[0] - 0.2, start_pose[1] - 0.1, start_pose[2] - 0.15,
                 target_params[0] - 0.1, target_params[1] - 0.1, target_params[2] - 0.1],
                [start_pose[0] + 0.2, start_pose[1] + 0.1, start_pose[2] + 0.15,
                 target_params[0] + 0.1, target_params[1] + 0.1, target_params[2] + 0.1],
            ],
            dtype=torch.float32, device=self.device,
        )

    def update_voxel_grid(self, depth_image, semantics, viewpoint):
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(viewpoint[3:], dtype=torch.float32, device=self.device)
        transform = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        coverage = self.voxel_grid.insert_depth_and_semantics(
            depth_image, semantics, transform
        )
        if coverage is not None:
            coverage = coverage.cpu().numpy()
        return coverage

    def update(self) -> np.array:
        r1, r2 = np.random.rand(2, self.n_particles) + 0.25
        r1 = torch.tensor(r1.reshape(self.n_particles, 1), dtype=torch.float32, device=self.device)
        r2 = torch.tensor(r2.reshape(self.n_particles, 1), dtype=torch.float32, device=self.device)

        self.V = (self.w * self.V
                  + self.c1 * r1 * (self.pbest - self.X)
                  + self.c2 * r2 * (self.gbest.repeat(self.n_particles, 1) - self.X))
        self.X = self.X + self.V

        for i in range(self.n_particles):
            for j in range(3):
                if self.X[i][j] >= self.camera_bounds[1][j]:
                    bouncing_force = self.camera_bounds[1][j] - self.X[i][j]
                    self.X[i][j] = self.camera_bounds[1][j]
                    self.V[i][j] = bouncing_force * self.bc
                if self.X[i][j] <= self.camera_bounds[0][j]:
                    bouncing_force = self.camera_bounds[0][j] - self.X[i][j]
                    self.X[i][j] = self.camera_bounds[0][j]
                    self.V[i][j] = bouncing_force * self.bc

        self.particle_trajectories.append(self.X.detach().cpu().numpy())

        obj = np.zeros(self.n_particles)
        i = 0
        for x in self.X:
            obj[i] = self.voxel_grid.compute_gain(x, self.target_params)[0]
            i += 1

        for i in range(len(self.obj_history)):
            self._push_obj_history(i, obj[i], self.X[i].detach().cpu().numpy())

        self.pbest = torch.tensor(
            np.array([min(particle, key=lambda x: x[0])[1] for particle in self.obj_history]),
            dtype=torch.float32, device=self.device,
        )
        self.pbest_obj = np.array(
            [min(particle, key=lambda x: x[0])[0] for particle in self.obj_history]
        )
        self.gbest = self.pbest[self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()
        return obj

    def _push_obj_history(self, i, utility, position):
        hist = self.obj_history[i]
        for k in range(self.recall - 1):
            hist[k][0] = hist[k + 1][0]
            hist[k][1] = hist[k + 1][1]
        hist[self.recall - 1][0] = utility
        hist[self.recall - 1][1] = position

    def pso_view(self) -> np.array:
        if self.init:
            self.init = False
            pos = self.gbest
            vp_utility = self.gbest_obj
        else:
            obj = self.update()
            pos = self.X[obj.argmin()]
            vp_utility = obj.min()

        quat = look_at_rotation(pos, self.target_params)
        viewpoint = np.zeros(7)
        viewpoint[:3] = pos.detach().cpu().numpy()
        viewpoint[3:] = quat.detach().cpu().numpy()
        return viewpoint, vp_utility

    def visualize(self):
        """No-op in ROS 2 (RViz visualizer removed)."""
        pass
