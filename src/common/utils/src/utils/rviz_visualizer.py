# ROS 2 node to visualize topics in rviz2

import struct
import numpy as np
from copy import deepcopy

from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Vector3, Point, Pose, PoseArray
from sensor_msgs.msg import PointCloud2, PointField, Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import ros2_node


def _array_to_pointcloud2(points_arr, stamp, frame_id):
    """Convert a structured numpy array (fields: x, y, z, rgb) to PointCloud2."""
    fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32,  count=1),
    ]
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = len(points_arr)
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(points_arr)
    msg.is_dense = True
    msg.data = points_arr.tobytes()
    return msg


class RvizVisualizer:
    def __init__(self):
        node = ros2_node.get_node()

        try:
            node.declare_parameter("world_frame_id", "world")
        except Exception:
            pass
        try:
            node.declare_parameter("camera_frame_id", "camera_link")
        except Exception:
            pass
        self.world_frame_id = (
            node.get_parameter("world_frame_id").get_parameter_value().string_value
            or "world"
        )
        self.camera_frame_id = (
            node.get_parameter("camera_frame_id").get_parameter_value().string_value
            or "camera_link"
        )

        self.view_samples_pub = node.create_publisher(MarkerArray, "view_samples", 1)
        self.viewpoint_pub    = node.create_publisher(Marker,      "viewpoint",     1)
        self.pc2_pub          = node.create_publisher(PointCloud2, "voxels",        1)
        self.rois_pub         = node.create_publisher(MarkerArray, "rois",          1)
        self.camera_bounds_pub   = node.create_publisher(Marker,      "camera_bounds",   1)
        self.semantic_mean_pub   = node.create_publisher(Marker,      "semantic_mean",   1)
        self.world_model_pub     = node.create_publisher(MarkerArray, "world_model/objects", 1)
        self.point_cloud_pub     = node.create_publisher(PointCloud2, "gt_point_cloud",     1)
        self.poses_with_covariance_pub = node.create_publisher(
            MarkerArray, "poses_with_covariance", 1
        )
        self.poses_pub       = node.create_publisher(PoseArray,   "pose_estimation/poses", 1)
        self.point_pub       = node.create_publisher(Marker,      "point",       1)
        self.class_ids_pub   = node.create_publisher(MarkerArray, "class_ids",   1)
        self.curve_pub       = node.create_publisher(Marker,      "curve",       1)
        self.points_pub      = node.create_publisher(Marker,      "points",      1)
        self.pred_points_pub = node.create_publisher(Marker,      "pred_points", 1)
        self.point_cloud_pub2 = node.create_publisher(PointCloud2, "true_point_cloud", 1)
        self.gain_image_pub  = node.create_publisher(Image,       "gain_image",  1)

    def _now(self):
        return ros2_node.get_node().get_clock().now().to_msg()

    def visualize_view_samples(self, view_samples: PoseArray) -> None:
        marker_array = MarkerArray()
        for i, pose in enumerate(view_samples.poses):
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = self._now()
            marker.ns = "view_samples"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale = Vector3(x=0.02, y=0.01, z=0.01)
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
            marker_array.markers.append(marker)
        self.view_samples_pub.publish(marker_array)

    def visualize_viewpoint(self, viewpoint: Pose) -> None:
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self._now()
        marker.ns = "viewpoint"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = viewpoint
        marker.scale = Vector3(x=0.02, y=0.01, z=0.01)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        self.viewpoint_pub.publish(marker)

    def visualize_voxels(self, points: np.array, semantics: np.array,
                         class_ids: np.array) -> None:
        assert points.shape[1] == 3
        color_occ = struct.unpack("I", struct.pack("BBBB", 60, 111, 2, 255))[0]
        color_0   = struct.unpack("I", struct.pack("BBBB", 0, 0, 255, 255))[0]
        color_1   = struct.unpack("I", struct.pack("BBBB", 0, 0, 255, 255))[0]
        points_dtype = np.dtype([("x", np.float32), ("y", np.float32),
                                  ("z", np.float32), ("rgb", np.uint32)])
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = color_occ
        for i in range(points.shape[0]):
            if class_ids[i] == 0:
                points_arr["rgb"][i] = color_0
            elif class_ids[i] == 1:
                points_arr["rgb"][i] = color_1
        voxel_points = _array_to_pointcloud2(points_arr, self._now(), self.world_frame_id)
        self.pc2_pub.publish(voxel_points)

    def visualize_rois(self, rois: PoseArray) -> None:
        marker_array = MarkerArray()
        for i, pose in enumerate(rois.poses):
            marker = Marker()
            marker.header.frame_id = self.world_frame_id
            marker.header.stamp = self._now()
            marker.ns = "rois"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
            marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            marker_array.markers.append(marker)
        self.rois_pub.publish(marker_array)

    def visualize_camera_bounds(self, bounds: np.array) -> None:
        lo = np.min(bounds, axis=0)
        hi = np.max(bounds, axis=0)
        corners = [
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face
            (4,5),(5,6),(6,7),(7,4),  # top face
            (0,4),(1,5),(2,6),(3,7),  # verticals
        ]
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self._now()
        marker.ns = "camera_bounds"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.005, y=0.0, z=0.0)
        marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
        for a, b in edges:
            marker.points.append(Point(x=float(corners[a][0]), y=float(corners[a][1]), z=float(corners[a][2])))
            marker.points.append(Point(x=float(corners[b][0]), y=float(corners[b][1]), z=float(corners[b][2])))
        self.camera_bounds_pub.publish(marker)

    def visualize_semantic_mean(self, mean: np.array) -> None:
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self._now()
        marker.ns = "semantic_mean"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=float(mean[0]), y=float(mean[1]),
                                     z=float(mean[2]))
        marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8)
        self.semantic_mean_pub.publish(marker)

    def visualize_wm(self, markers: MarkerArray) -> None:
        self.world_model_pub.publish(markers)

    def _make_rgb_cloud(self, points: np.array, color: np.array, pub):
        assert points.shape[1] == 3
        assert color.shape[1] == 3
        color = (color * 255).astype(np.uint8)
        points_dtype = np.dtype([("x", np.float32), ("y", np.float32),
                                  ("z", np.float32), ("rgb", np.uint32)])
        points_arr = np.empty(points.shape[0], dtype=points_dtype)
        points_arr["x"] = points[:, 0]
        points_arr["y"] = points[:, 1]
        points_arr["z"] = points[:, 2]
        points_arr["rgb"] = np.array(
            [struct.unpack("I", struct.pack("BBBB", *color[i, ::-1], 255))[0]
             for i in range(color.shape[0])]
        )
        cloud = _array_to_pointcloud2(points_arr, self._now(), "world")
        pub.publish(cloud)

    def visualize_point_cloud(self, points: np.array, color: np.array) -> None:
        self._make_rgb_cloud(points, color, self.point_cloud_pub)

    def visualize_gt_point_cloud(self, points: np.array, color: np.array) -> None:
        self._make_rgb_cloud(points, color, self.point_cloud_pub2)

    def visualize_curve(self, x, y, z):
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self._now()
        marker.ns = "curve"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        for i in range(len(x)):
            marker.points.append(Point(x=float(x[i]), y=float(y[i]), z=float(z[i])))
        self.curve_pub.publish(marker)

    def visualize_point(self, point: np.array):
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self._now()
        marker.ns = "point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=float(point[0]), y=float(point[1]),
                                     z=float(point[2]))
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        self.point_pub.publish(marker)

    def visualize_points(self, points: np.ndarray):
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self._now()
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.01, y=0.01, z=0.01)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        for point in points:
            marker.points.append(Point(x=float(point[0]), y=float(point[1]),
                                       z=float(point[2])))
        self.points_pub.publish(marker)

    def visualize_pred_points(self, points: np.ndarray):
        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self._now()
        marker.ns = "pred_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.02, y=0.02, z=0.02)
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
        for point in points:
            marker.points.append(Point(x=float(point[0]), y=float(point[1]),
                                       z=float(point[2])))
        self.pred_points_pub.publish(marker)

    def visualize_gain_image(self, image: np.ndarray):
        bridge = CvBridge()
        image = (image * 255).astype(np.uint8)[:, :, ::-1]
        image_msg = bridge.cv2_to_imgmsg(image, encoding="passthrough")
        image_msg.header.frame_id = self.camera_frame_id
        image_msg.header.stamp = self._now()
        self.gain_image_pub.publish(image_msg)
