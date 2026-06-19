#!/usr/bin/env python3
"""Debug a SINGLE compute_ik call and print the raw error_code + solver info.

Tries several ik_link_name candidates and both avoid_collisions settings so we
can see exactly why IK was refused.
"""
import numpy as np
import ros2_node
from utils.py_utils import look_at_rotation

import rclpy
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped

TARGET = np.array([0.5, -0.4, 1.0])
GROUP = "manipulator"

# A pose we KNOW the arm reached in the motion test: (0.5, -0.6, 1.05) REACH
CAM = np.array([0.5, -0.6, 1.05])

def quat_xyzw(cam):
    w, x, y, z = look_at_rotation(np.asarray(cam, float), TARGET)  # [w,x,y,z]
    return x, y, z, w

def try_ik(node, cli, link, avoid_coll):
    qx, qy, qz, qw = quat_xyzw(CAM)
    req = GetPositionIK.Request()
    req.ik_request.group_name = GROUP
    if link:
        req.ik_request.ik_link_name = link
    req.ik_request.pose_stamped = PoseStamped()
    req.ik_request.pose_stamped.header.frame_id = "world"
    p = req.ik_request.pose_stamped.pose
    p.position.x, p.position.y, p.position.z = map(float, CAM)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
    req.ik_request.timeout.sec = 2
    req.ik_request.avoid_collisions = avoid_coll
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=8.0)
    if not fut.done():
        return "NO RESPONSE (future not done)"
    res = fut.result()
    if res is None:
        return "RESULT IS NONE"
    n_joints = len(res.solution.joint_state.position)
    return f"error_code={res.error_code.val}  joints_returned={n_joints}"

def main():
    ros2_node.init()
    node = ros2_node.get_node()
    cli = node.create_client(GetPositionIK, "/compute_ik")
    if not cli.wait_for_service(timeout_sec=10.0):
        print("no /compute_ik"); return
    print(f"\nDebugging IK for cam={CAM} -> target={TARGET}")
    print(f"quat (x,y,z,w) = {quat_xyzw(CAM)}\n")
    for link in ["camera_color_frame", "tool0", "wrist_3_link", "flange", ""]:
        for ac in [True, False]:
            label = link if link else "(default tip)"
            print(f"  link={label:22s} avoid_collisions={ac!s:5s} -> {try_ik(node, cli, link, ac)}")
    print("\n(error_code: 1=SUCCESS, -31=NO_IK_SOLUTION, -10..-12=collision/frame issues)")

if __name__ == "__main__":
    main()
