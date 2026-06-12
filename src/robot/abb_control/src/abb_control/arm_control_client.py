#!/usr/bin/env python3
import time
import random
from geometry_msgs.msg import Pose, Point, Quaternion
from abb_interfaces.srv import ArmGoal
import ros2_node


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

    def move_arm_to_pose(self, pose):
        node = ros2_node.get_node()
        # Ensure the spin thread is alive before sending — it may have died during
        # the long planning window if the rclpy context was inadvertently shut down.
        import rclpy
        spin_thread = ros2_node._spin_thread
        if spin_thread is None or not spin_thread.is_alive():
            print(f"[ArmControlClient] Spin thread dead (rclpy.ok={rclpy.ok()}), restarting...")
            if rclpy.ok() and ros2_node._executor is not None:
                import threading
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
            deadline = time.monotonic() + 60.0
            while not future.done():
                if time.monotonic() > deadline:
                    print("[ArmControlClient] Service call timed out after 60s")
                    return False
                time.sleep(0.01)
            result = future.result()
            if result and result.success:
                node.get_logger().info("[ArmControlClient] Arm arrived at pose")
            else:
                node.get_logger().error("[ArmControlClient] Arm motion failed")
            return result.success if result else False
        except Exception as e:
            print(f"[ArmControlClient] Service call failed: {e}")
            return False
