#!/usr/bin/env python3
import time
import tf2_ros
from geometry_msgs.msg import TransformStamped
import ros2_node

try:
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState, GetModelState
    _GAZEBO_MSGS = True
except ImportError:
    _GAZEBO_MSGS = False
    print("[ArmControlGazebo] gazebo_msgs not found — Gazebo state services unavailable. "
          "Install with: sudo apt install ros-jazzy-gazebo-msgs")


class ArmControlGazebo:
    def __init__(self):
        node = ros2_node.get_node()
        self.br = tf2_ros.TransformBroadcaster(node)
        self.arm_name = "abb_l515"
        self.arm_frame = "base_link"
        self.ref_frame = "world"

        if _GAZEBO_MSGS:
            self.get_state_client = node.create_client(
                GetModelState, "/gazebo/get_model_state"
            )
            self.set_state_client = node.create_client(
                SetModelState, "/gazebo/set_model_state"
            )
            node.get_logger().info(
                "[ArmControlGazebo] Waiting for Gazebo state services"
            )
            self.get_state_client.wait_for_service()
            self.set_state_client.wait_for_service()
        else:
            self.get_state_client = None
            self.set_state_client = None

    def get_agent_pose(self):
        if not _GAZEBO_MSGS or self.get_state_client is None:
            print("[ArmControlGazebo] get_agent_pose unavailable (gazebo_msgs missing)")
            return None
        req = GetModelState.Request()
        req.model_name = self.arm_name
        req.relative_entity_name = self.ref_frame
        future = self.get_state_client.call_async(req)
        while not future.done():
            time.sleep(0.01)
        result = future.result()
        if result:
            return result
        print("[ArmControlGazebo] Failed to get model state")

    def move_agent_to_pose(self, pose):
        if not _GAZEBO_MSGS or self.set_state_client is None:
            print("[ArmControlGazebo] move_agent_to_pose unavailable (gazebo_msgs missing)")
            return
        req = SetModelState.Request()
        req.model_state = ModelState()
        req.model_state.model_name = self.arm_name
        req.model_state.pose = pose
        req.model_state.reference_frame = self.ref_frame
        try:
            future = self.set_state_client.call_async(req)
            while not future.done():
                time.sleep(0.01)
        except Exception as e:
            print("[ArmControlGazebo] Service call to set_model_state failed:", e)

    def broadcast(self, pose):
        node = ros2_node.get_node()
        t = TransformStamped()
        t.header.stamp = node.get_clock().now().to_msg()
        t.header.frame_id = self.ref_frame
        t.child_frame_id = self.arm_frame
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self.br.sendTransform(t)
