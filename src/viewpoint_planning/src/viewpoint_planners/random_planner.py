import torch
import numpy as np

from scene_representation.voxel_grid import VoxelGrid
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import transform_from_rotation_translation

from viewpoint_planners.planner_eval_mixin import PlannerEvalMixin, init_eval_state
from viewpoint_planners.fair_comparison_config import (
    GRID_SIZE as FC_GRID_SIZE,
    VOXEL_SIZE as FC_VOXEL_SIZE,
)


class _NoOpVisualizer:
    """ROS 2 stub — RViz visualization calls are silently ignored."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class RandomPlanner(PlannerEvalMixin):
    """
    Random viewpoint sampler baseline.
    ROS 2 Jazzy compatible — RvizVisualizer replaced with no-op stub.
    Random sampling strategy is UNCHANGED.
    """

    def __init__(
        self,
        start_pose: np.array,
        mesh_coordinates: np.array = None,
        mesh_tree=None,
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
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.random_params(start_pose, target_params)
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
        self.view_sampler = ViewpointSampler(num_samples)
        self.viewpoint = start_pose
        self.target_position = target_params
        self.rviz_visualizer = _NoOpVisualizer()  # ROS 2: no-op stub

        self.mesh_coordinates = mesh_coordinates
        self.mesh_tree = mesh_tree
        init_eval_state(self)

    def random_params(self, start_pose: np.array, target_params: np.array) -> None:
        self.target_params = torch.tensor(
            target_params, dtype=torch.float32, device=self.device,
        )
        self.camera_bounds = np.array(
            [
                [start_pose[0] - 0.2, start_pose[1] - 0.2, start_pose[2] - 0.25,
                 target_params[0] - 0.2, target_params[1] - 0.2, target_params[2] - 0.2],
                [start_pose[0] + 0.2, start_pose[1] + 0.2, start_pose[2] + 0.25,
                 target_params[0] + 0.2, target_params[1] + 0.2, target_params[2] + 0.2],
            ]
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

    def random_view(self) -> np.array:
        view_samples = self.view_sampler.random_neighbour_sampler(
            self.viewpoint[:3],
            self.target_position,
            camera_limits=self.camera_bounds[:, :3],
            target_limits=self.camera_bounds[:, 3:],
        )
        random_index = np.random.randint(self.num_samples)
        viewpoint = view_samples[random_index, :7]
        self.target_position = view_samples[random_index, 7:]
        self.viewpoint = viewpoint
        return self.viewpoint, 0.0, 1

    def visualize(self):
        """No-op in ROS 2 (RViz visualizer removed)."""
        pass
