"""
Perceiver: get RGB-D from camera, do semantic segmentation, estimate target
node 3D position.

Extended with extract_node_position() to support Burusa-style Kalman-filter
node tracking (Section IV-A of Burusa et al. ICRA 2024).
"""

import rospy
import cv2
import torch
import numpy as np

from perception.realsense_ros_capturer import RealsenseROSCapturer


class Perceiver:
    """
    Gets data from the camera and performs semantic segmentation and pose estimation.
    """

    def __init__(self):
        self.capturer = RealsenseROSCapturer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_camera_info(self):
        camera_info, _, _ = self.capturer.get_frames()
        return camera_info

    def run(self):
        # Get data from camera
        camera_info, color_output, depth_output = self.capturer.get_frames()
        color_image = color_output["color_image"]
        depth_image = depth_output["depth_image"]
        points = depth_output["points"]
        # Return if no data
        if camera_info is None or color_image is None:
            rospy.logwarn("[Perceiver] Perception paused. No data from camera.")
            return
        # Color-based segmentation
        # Note: Only for the toy example. Replace with an object-detection
        # network in practice.
        segmentation_mask = self.color_segmentation(color_image)
        # Get semantics
        semantics = self.assign_semantics(camera_info, segmentation_mask)
        return depth_image, points, semantics

    def color_segmentation(self, color_image: np.array) -> np.array:
        """
        Perform color segmentation on the input image using OpenCV
        :param color_image: input color image
        :return: segmentation mask
        """
        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # Define range of red color in HSV
        lower_color = np.array([0, 50, 50])
        upper_color = np.array([10, 255, 255])
        # Threshold the HSV image to get only white colors
        segmentation_mask = cv2.inRange(hsv_image, lower_color, upper_color)
        return segmentation_mask

    def assign_semantics(self, camera_info, segmentation_mask) -> torch.tensor:
        """
        Assign the confidence scores and labels to the pixels in the image
        :param camera_info: camera information
        :param segmentation_mask: segmentation mask [H x W]
        :return: semantic confidence scores and labels [H x W x 2]
        """
        # Image size
        image_size = (camera_info.height, camera_info.width)
        # Create a mask that is log odds 0.9 if there's a semantic value and
        # log odds of 0.4 otherwise
        occupied_odds = self.log_odds(0.9)
        free_odds = self.log_odds(0.4)
        # Initialize the label mask as free
        score_mask = free_odds * torch.ones(
            image_size, dtype=torch.float32, device=self.device
        )
        # Initialize the label mask as background
        label_mask = -1 * torch.ones(
            image_size, dtype=torch.float32, device=self.device
        )
        # Assign the semantic labels
        score_mask[segmentation_mask > 0] = occupied_odds
        label_mask[segmentation_mask > 0] = 0
        semantics = torch.stack((score_mask, label_mask), dim=-1)
        return semantics

    def log_odds(self, p):
        return np.log(p / (1 - p))

    # ================================================================
    # NEW: 3D node position extraction for Kalman-filter tracking.
    # Burusa et al. (ICRA 2024), Section IV-A:
    #   "For each detected node, its instance mask was applied to the
    #    aligned point cloud and the points that belonged to the detected
    #    node were extracted. The 3D position of the node was estimated as
    #    the mean position of the extracted points. The uncertainty in the
    #    3D position was estimated as the variance of the extracted points
    #    along the viewing direction of the camera."
    # ================================================================
    def extract_node_position(
        self,
        points: np.array,
        segmentation_mask: np.array,
        camera_pose: np.array,
    ):
        """Estimate 3D centroid + uncertainty of the segmented target node
        in this view.

        Parameters
        ----------
        points : np.ndarray shape (H, W, 3)
            Aligned point cloud (one 3D point per pixel) in the world frame.
            Pixels with no depth are typically NaN.
        segmentation_mask : np.ndarray shape (H, W)
            Binary mask from `color_segmentation` (> 0 means target).
        camera_pose : np.ndarray shape (7,)
            Current camera pose [x, y, z, qx, qy, qz, qw]. The camera
            position is used to define the viewing direction along which
            the uncertainty is computed (Burusa IV-A).

        Returns
        -------
        position : np.ndarray shape (3,) or None
            Mean 3D position of the segmented target points, or None if
            the target is not visible (segmentation empty / all NaN).
        uncertainty : float or None
            Per-axis standard deviation along the camera viewing direction
            (m). None if position is None.
        detected : bool
            True if the target was detected in this view (at least one
            valid segmented point with non-NaN depth).
        """
        # 1) Reshape and gather only segmented pixels with valid depth.
        if points is None or points.size == 0:
            return None, None, False

        mask = (segmentation_mask > 0)
        if not mask.any():
            return None, None, False

        # `points` may be (H, W, 3) or (N, 3) -- accept both.
        if points.ndim == 3:
            target_pts = points[mask]                 # (M, 3)
        else:
            # Already flattened. We assume row-major flatten of the mask.
            target_pts = points[mask.flatten()]

        # 2) Drop NaN/inf rows (RealSense depth often has invalid pixels).
        finite = np.isfinite(target_pts).all(axis=1)
        target_pts = target_pts[finite]
        if target_pts.shape[0] == 0:
            return None, None, False

        # 3) Mean position = Burusa's per-view 3D node estimate (Section IV-A).
        position = target_pts.mean(axis=0)

        # 4) Uncertainty = variance along the camera viewing direction
        #    (Burusa Section IV-A: "the main source of error in 3D position
        #     due to depth noise"). Project (pts - mean) onto view direction
        #    and take its standard deviation.
        cam_pos = np.asarray(camera_pose[:3], dtype=np.float64)
        view_dir = position - cam_pos
        view_dir_norm = np.linalg.norm(view_dir)
        if view_dir_norm < 1e-6:
            # Degenerate (camera coincides with centroid): fall back to
            # isotropic std across all axes.
            uncertainty = float(np.mean(target_pts.std(axis=0)))
        else:
            view_dir = view_dir / view_dir_norm
            projected = (target_pts - position) @ view_dir   # (M,)
            uncertainty = float(np.std(projected))

        # Safety floor (m): too small a value will make Kalman update
        # essentially overwrite the prior with the measurement.
        uncertainty = max(uncertainty, 1e-3)

        return position.astype(np.float64), uncertainty, True

    def run_with_node_position(self, camera_pose: np.array):
        """Convenience wrapper: returns the same as `run()` plus the
        per-view 3D node position + uncertainty for Kalman tracking.

        Use this in place of `run()` when you want Kalman-filter tracking
        active. Keeps `run()` backward-compatible for any code that does
        not need the tracking output.
        """
        camera_info, color_output, depth_output = self.capturer.get_frames()
        color_image = color_output["color_image"]
        depth_image = depth_output["depth_image"]
        points = depth_output["points"]
        if camera_info is None or color_image is None:
            rospy.logwarn("[Perceiver] Perception paused. No data from camera.")
            return None
        segmentation_mask = self.color_segmentation(color_image)
        semantics = self.assign_semantics(camera_info, segmentation_mask)

        position, uncertainty, detected = self.extract_node_position(
            points, segmentation_mask, camera_pose
        )

        return {
            "depth_image": depth_image,
            "points": points,
            "semantics": semantics,
            "node_position": position,         # np.ndarray (3,) or None
            "node_uncertainty": uncertainty,   # float or None
            "node_detected": detected,         # bool
        }
