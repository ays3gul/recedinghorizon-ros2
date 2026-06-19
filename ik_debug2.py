#!/usr/bin/env python3
"""Debug #2: error_code -15 = FRAME_TRANSFORM_FAILURE.

Isolate the frame problem. Try:
  - frame_id = world  vs  base_link
  - with and without a fresh header.stamp
  - also fetch the world->base_link TF so we can convert the pose ourselves
"""
import numpy as np
import ros2_node
from utils.py_utils import look_at_rotation

import rclpy
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import time

TARGET = np.array([0.5, -0.4, 1.0])
GROUP = "manipulator"
CAM = np.array([0.5, -0.6, 1.05])   # reached in the motion test

def quat_xyzw(cam):
    w, x, y, z = look_at_rotation(np.asarray(cam, float), TARGET)
    return x, y, z, w

def try_ik(node, cli, frame_id, set_stamp, pos, link="tool0"):
    qx, qy, qz, qw = quat_xyzw(CAM)
    req = GetPositionIK.Request()
    req.ik_request.group_name = GROUP
    req.ik_request.ik_link_name = link
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    if set_stamp:
        ps.header.stamp = node.get_clock().now().to_msg()
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos)
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
    req.ik_request.pose_stamped = ps
    req.ik_request.timeout.sec = 2
    req.ik_request.avoid_collisions = False
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=8.0)
    if not fut.done() or fut.result() is None:
        return "NO RESPONSE"
    return f"error_code={fut.result().error_code.val}"

def main():
    ros2_node.init()
    node = ros2_node.get_node()

    # --- grab world->base_link TF ---
    buf = Buffer()
    TransformListener(buf, node)
    time.sleep(2.0)  # let TF fill
    tf_str = "??"
    base_pos = None
    for parent, child in [("base_link", "world"), ("world", "base_link")]:
        try:
            t = buf.lookup_transform(parent, child, rclpy.time.Time())
            tr = t.transform.translation
            ro = t.transform.rotation
            tf_str = f"{parent}<-{child}: t=({tr.x:.3f},{tr.y:.3f},{tr.z:.3f}) q=({ro.x:.3f},{ro.y:.3f},{ro.z:.3f},{ro.w:.3f})"
            if parent == "base_link":
                # convert CAM (world) into base_link coords (assume no rotation; verify from q)
                base_pos = np.array([CAM[0]+tr.x, CAM[1]+tr.y, CAM[2]+tr.z])
            break
        except Exception as e:
            tf_str = f"lookup failed: {e}"
    print("\nTF:", tf_str)
    if base_pos is not None:
        print(f"CAM in base_link (translation-only): {base_pos}")

    cli = node.create_client(GetPositionIK, "/compute_ik")
    cli.wait_for_service(timeout_sec=10.0)

    print("\n--- frame / stamp sweep (link=tool0, pose in WORLD coords) ---")
    print(f"  world,    stamp=no  -> {try_ik(node, cli, 'world', False, CAM)}")
    print(f"  world,    stamp=yes -> {try_ik(node, cli, 'world', True,  CAM)}")
    print(f"  base_link,stamp=no  -> {try_ik(node, cli, 'base_link', False, CAM)}")
    print(f"  base_link,stamp=yes -> {try_ik(node, cli, 'base_link', True,  CAM)}")
    print(f"  (empty),  stamp=no  -> {try_ik(node, cli, '', False, CAM)}")

    if base_pos is not None:
        print("\n--- pose CONVERTED to base_link coords, frame_id=base_link ---")
        print(f"  base_link,stamp=no  -> {try_ik(node, cli, 'base_link', False, base_pos)}")
        print(f"  base_link,stamp=yes -> {try_ik(node, cli, 'base_link', True,  base_pos)}")

    print("\n(-15 = FRAME_TRANSFORM_FAILURE, 1 = SUCCESS, -31 = NO_IK_SOLUTION)")

if __name__ == "__main__":
    main()
