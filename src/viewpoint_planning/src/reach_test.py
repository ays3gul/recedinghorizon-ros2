#!/usr/bin/env python3
"""
Reach-mapping test for the UR5e + L515 setup.

Sends a grid of 'look-at-the-bunny' poses to move_arm_to_pose and records
which ones the arm can actually reach. The reachable set tells us how to set
robot_reach_bounds so the camera can orbit the object instead of staring at
it from one side.

Run AFTER move_group + arm_control are up (same way you run the experiment),
e.g.:
    cd ~/Desktop/RecedingHorizon
    python3 reach_test.py

It does NOT spawn occluders or run the planner — it only probes reachability.
"""

import numpy as np

# --- reuse your existing client + pose helper so frames/conventions match ---
from abb_control.arm_control_client import ArmControlClient
from utils.py_utils import numpy_to_pose
import ros2_node  # noqa: F401  (ensures the rclpy node/spin thread is alive)


# Bunny world position (matches bunny.world / target_position)
TARGET = np.array([0.5, -0.4, 1.0])


def look_at_quaternion(cam_pos, target):
    """Quaternion (x, y, z, w) so the camera at cam_pos looks toward target.

    This mirrors how the planner orients the camera (look-at toward the ROI),
    so the reach test exercises realistic orientations, not just positions.
    Convention: camera 'forward' is +Z of camera_color_frame (REP-145 optical
    parent). If your look_at in the planner uses a different forward axis,
    tell me and I'll match it exactly — but for a reach probe this is fine.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    fwd = target - cam_pos
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    fwd /= n
    # Build a rotation whose +Z axis points along fwd.
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, fwd)) > 0.95:           # fwd nearly vertical -> pick other up
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    true_up = np.cross(fwd, right)
    R = np.column_stack([right, true_up, fwd])   # columns = camera x,y,z in world
    # rotation matrix -> quaternion (x,y,z,w)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def make_pose(cam_pos):
    """7-vector [x,y,z, qx,qy,qz,qw] expected by numpy_to_pose."""
    q = look_at_quaternion(cam_pos, TARGET)
    return np.array([cam_pos[0], cam_pos[1], cam_pos[2], q[0], q[1], q[2], q[3]])


def main():
    arm = ArmControlClient()

    # Probe grid: orbit the bunny on multiple sides at a ~0.30-0.40 m radius.
    # We deliberately go BEYOND the current bounds (esp. in -Y, past the bunny)
    # to find the true reachable envelope.
    xs = [0.35, 0.45, 0.50, 0.55, 0.65]
    ys = [-0.70, -0.60, -0.50, -0.40, -0.25, -0.10, 0.05]
    zs = [0.95, 1.05, 1.15, 1.25]

    reachable = []
    failed = []
    total = len(xs) * len(ys) * len(zs)
    idx = 0

    print(f"\n[reach_test] Probing {total} look-at poses around bunny at {TARGET}\n")

    for x in xs:
        for y in ys:
            for z in zs:
                idx += 1
                cam = np.array([x, y, z])
                # skip poses essentially on top of / inside the object
                if np.linalg.norm(cam - TARGET) < 0.18:
                    print(f"[{idx}/{total}] ({x:+.2f},{y:+.2f},{z:+.2f}) SKIP (too close)")
                    continue
                pose = make_pose(cam)
                ok = arm.move_arm_to_pose(numpy_to_pose(pose))
                tag = "REACH " if ok else "  FAIL"
                print(f"[{idx}/{total}] ({x:+.2f},{y:+.2f},{z:+.2f}) -> {tag}")
                (reachable if ok else failed).append(cam)

    reachable = np.array(reachable) if reachable else np.empty((0, 3))
    print("\n" + "=" * 60)
    print(f"  REACHABLE: {len(reachable)} / {total}")
    print("=" * 60)
    if len(reachable):
        lo = reachable.min(axis=0)
        hi = reachable.max(axis=0)
        print(f"  Reachable bounding box (look-at poses):")
        print(f"    x: [{lo[0]:.3f}, {hi[0]:.3f}]")
        print(f"    y: [{lo[1]:.3f}, {hi[1]:.3f}]")
        print(f"    z: [{lo[2]:.3f}, {hi[2]:.3f}]")
        print(f"\n  Suggested robot_reach_bounds (shrunk 2cm inward for safety):")
        print(f"    np.array([[{lo[0]+0.02:.2f}, {lo[1]+0.02:.2f}, {lo[2]+0.02:.2f}],")
        print(f"              [{hi[0]-0.02:.2f}, {hi[1]-0.02:.2f}, {hi[2]-0.02:.2f}]])")
    else:
        print("  Nothing reachable — check that move_group/arm_control are up.")
    print()


if __name__ == "__main__":
    main()
