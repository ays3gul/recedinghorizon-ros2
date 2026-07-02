#!/usr/bin/env python3
import time
import random
import math
import threading
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Point, Quaternion
from abb_interfaces.srv import ArmGoal
import ros2_node

try:
    import tf2_ros
    _TF2_AVAILABLE = True
except ImportError:
    _TF2_AVAILABLE = False


def RandomPoseGenerator(minx, maxx, miny, maxy, minz, maxz):
    return Pose(
        position=Point(x=random.uniform(minx, maxx),
                       y=random.uniform(miny, maxy),
                       z=random.uniform(minz, maxz)),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


class ArmControlClient:
    def __init__(self):
        node = ros2_node.get_node()
        node.get_logger().info("[ArmControlClient] Initializing arm control")
        self.client = node.create_client(ArmGoal, "move_arm_to_pose")
        self.minx = 0.40
        self.maxx = 0.80
        self.miny = -0.40
        self.maxy = 0.40
        self.minz = 1.00
        self.maxz = 1.40

        # TF buffer for pose verification fallback
        self._tf_buffer = None
        self._tf_listener = None
        if _TF2_AVAILABLE:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        node.get_logger().info(
            "[ArmControlClient] Waiting for move_arm_to_pose service..."
        )
        if not self.client.wait_for_service(timeout_sec=10.0):
            node.get_logger().warn(
                "[ArmControlClient] move_arm_to_pose service not available after 10s. "
                "Is the arm controller running?"
            )
        else:
            node.get_logger().info("[ArmControlClient] Arm control ready")

    def _arm_at_pose(self, pose, pos_tol=0.08):
        """Return True if camera_color_frame is within pos_tol of the requested pose."""
        if self._tf_buffer is None:
            return False
        try:
            from rclpy.time import Time
            t = self._tf_buffer.lookup_transform(
                'world', 'camera_color_frame', Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
            dx = abs(t.transform.translation.x - pose.position.x)
            dy = abs(t.transform.translation.y - pose.position.y)
            dz = abs(t.transform.translation.z - pose.position.z)
            return dx < pos_tol and dy < pos_tol and dz < pos_tol
        except Exception:
            return False

    def get_camera_pose(self):
        """Actual camera_color_frame pose in world as [x,y,z, w,qx,qy,qz]
        (wxyz quat = planner/numpy_to_pose convention). None if TF unavailable.
        Used to integrate the voxel grid at the pose the arm ACTUALLY reached,
        not the commanded one (reduces multi-view smear from execution error)."""
        if self._tf_buffer is None:
            return None
        try:
            from rclpy.time import Time
            t = self._tf_buffer.lookup_transform(
                'world', 'camera_color_frame', Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            tr = t.transform.translation
            q = t.transform.rotation
            return np.array([tr.x, tr.y, tr.z, q.w, q.x, q.y, q.z])
        except Exception:
            return None

    def move_arm_to_pose(self, pose):
        node = ros2_node.get_node()
        spin_thread = ros2_node._spin_thread
        if spin_thread is None or not spin_thread.is_alive():
            print(f"[ArmControlClient] Spin thread dead (rclpy.ok={rclpy.ok()}), restarting...")
            if rclpy.ok() and ros2_node._executor is not None:
                ros2_node._spin_thread = threading.Thread(
                    target=ros2_node._executor.spin, daemon=True
                )
                ros2_node._spin_thread.start()
                print("[ArmControlClient] Spin thread restarted")
        try:
            node.get_logger().info(
                f"[ArmControlClient] Requesting move to pose: {pose}"
            )
            req = ArmGoal.Request()
            req.goal_pose = pose
            future = self.client.call_async(req)
            deadline = time.monotonic() + 180.0
            while not future.done():
                if time.monotonic() > deadline:
                    print("[ArmControlClient] Service call timed out after 180s")
                    return False
                time.sleep(0.01)
            result = future.result()
            if result and result.success:
                node.get_logger().info("[ArmControlClient] Arm arrived at pose")
                return True

            # MoveIt2 execute() has a race condition where it reports failure
            # even though the trajectory succeeded. Verify via TF before giving up.
            time.sleep(1.5)
            if self._arm_at_pose(pose):
                node.get_logger().info(
                    "[ArmControlClient] Arm arrived at pose (verified via TF)"
                )
                return True

            node.get_logger().error("[ArmControlClient] Arm motion failed")
            return False
        except Exception as e:
            print(f"[ArmControlClient] Service call failed: {e}")
            return False
