import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
import ros2_node


class RealsenseROSCapturer:
    """
    Gets the color and depth frames of a realsense RGB-D camera from ROS 2 (D4XX, D5XX).
    """

    def __init__(self):
        node = ros2_node.get_node()
        self._bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.points = None
        self.camera_info = None
        self._new_frame_event = threading.Event()

        self.use_sim = node.get_parameter("use_sim_time").get_parameter_value().bool_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )

        node.create_subscription(Image, "/camera/color/image_rect_color",
                                 self.color_callback, qos)
        node.create_subscription(CameraInfo,
                                 "/camera/aligned_depth_to_color/camera_info",
                                 self.info_callback, 1)
        node.create_subscription(Image,
                                 "/camera/aligned_depth_to_color/image_raw",
                                 self.depth_callback, 1)
        node.create_subscription(PointCloud2,
                                 "/camera/depth_registered/points",
                                 self.points_callback,
                                 QoSProfile(
                                     reliability=ReliabilityPolicy.BEST_EFFORT,
                                     history=HistoryPolicy.KEEP_LAST,
                                     depth=1,
                                 ))

    def color_callback(self, msg):
        data = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.color_image = data[:, :, ::-1]  # BGR→RGB
        self._new_frame_event.set()

    def depth_callback(self, msg):
        data = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(
            "float32"
        )
        # Gazebo sends 32FC1 (float32, already in metres).
        # Real RealSense cameras send 16UC1 (uint16, millimetres → divide by 1000).
        if msg.encoding in ("16UC1", "16uc1"):
            data = data / 1000.0
        self.depth_image = data

    def points_callback(self, msg):
        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z"),
                                    skip_nans=False)
        data = pts.reshape(msg.height, msg.width, 3).astype(np.float32)
        # NOTE: y-offset removed — it caused a depth/point-cloud mismatch
        # that shifted reconstructed voxels off the mesh surface.
        self.points = data

    def info_callback(self, msg):
        self.camera_info = msg

    def get_frames(self, wait_for_new=True, timeout=3.0):
        if wait_for_new:
            self._new_frame_event.clear()
            self._new_frame_event.wait(timeout=timeout)
        # Wait an extra second for camera_info if it hasn't arrived yet — camera_info
        # is published on a separate topic and can lag behind color frames.
        if self.camera_info is None:
            import time as _time
            deadline = _time.monotonic() + 5.0
            while self.camera_info is None and _time.monotonic() < deadline:
                _time.sleep(0.05)
        color_output = {"color_image": self.color_image}
        depth_output = {"depth_image": self.depth_image, "points": self.points}
        return self.camera_info, color_output, depth_output
