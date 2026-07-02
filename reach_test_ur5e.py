#!/usr/bin/env python3
"""
reach_test_ur5e.py — UR5e reachability grid test for current bunny position.

Sweeps a dense x/y/z grid of look-at-bunny poses and records which ones
the arm can actually reach. Prints the resulting bounding box to use as
CAMERA_BOUNDS_HALFWIDTHS in fair_comparison_config.py.

Run with robot stack up (move_group + arm_control_node):
    cd ~/Desktop/RecedingHorizon
    conda activate rh_nbv_ros2
    source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
    python3 reach_test_ur5e.py
"""
import csv
import time
import numpy as np

import ros2_node
from abb_control.arm_control_client import ArmControlClient
from utils.py_utils import numpy_to_pose, look_at_rotation

# Current bunny position (matches fair_comparison_config.py)
TARGET = np.array([0.5, -0.25, 1.1])

# Grid to probe — focused on the region that is both UR5e-reachable AND
# D455-valid for the object now at x=0.3. UR5e reachable camera x tops out
# ~0.40 (base at world origin), so probing x>0.45 only wastes ~1.5 s/pose on
# guaranteed failures. Widen later only if the envelope hits a grid edge.
XS = np.arange(0.10, 0.45, 0.05)
YS = np.arange(-0.25, 0.45, 0.05)
ZS = np.arange(0.85, 1.45, 0.05)

MIN_DIST = 0.40   # D455 minimum reliable depth — closer viewpoints are unusable
MAX_DIST = 0.65   # skip poses too far (camera won't see bunny well)


def make_pose(cam_pos):
    q = look_at_rotation(cam_pos.astype(float), TARGET.astype(float))
    return np.concatenate((cam_pos.astype(float), q))


def main():
    ros2_node.init("reach_test_ur5e")
    arm = ArmControlClient()

    poses = []
    for x in XS:
        for y in YS:
            for z in ZS:
                cam = np.array([x, y, z])
                d = np.linalg.norm(cam - TARGET)
                if MIN_DIST <= d <= MAX_DIST:
                    poses.append(cam)

    total = len(poses)
    print(f"\n[reach_test_ur5e] TARGET={TARGET.tolist()}")
    print(f"Probing {total} poses (grid step 5cm, dist [{MIN_DIST},{MAX_DIST}]m from bunny)\n")

    reachable = []
    failed = []

    with open("reach_test_ur5e_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "reachable"])

        for i, cam in enumerate(poses):
            ok = arm.move_arm_to_pose(numpy_to_pose(make_pose(cam)))
            tag = "REACH" if ok else " FAIL"
            print(f"[{i+1:4d}/{total}] ({cam[0]:+.2f},{cam[1]:+.2f},{cam[2]:+.2f}) -> {tag}")
            w.writerow([f"{cam[0]:.2f}", f"{cam[1]:.2f}", f"{cam[2]:.2f}", int(bool(ok))])
            f.flush()
            (reachable if ok else failed).append(cam)

    print("\n" + "=" * 60)
    print(f"  REACHABLE: {len(reachable)} / {total}")
    print("=" * 60)

    if reachable:
        r = np.array(reachable)
        lo = r.min(axis=0)
        hi = r.max(axis=0)
        center = (lo + hi) / 2
        half = (hi - lo) / 2

        print(f"\n  Reachable bounding box:")
        print(f"    x: [{lo[0]:.3f}, {hi[0]:.3f}]")
        print(f"    y: [{lo[1]:.3f}, {hi[1]:.3f}]")
        print(f"    z: [{lo[2]:.3f}, {hi[2]:.3f}]")
        print(f"\n  Box center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"\n  Suggested CAMERA_BOUNDS_HALFWIDTHS (5cm safety margin):")
        print(f"    np.array([{half[0]-0.05:.2f}, {half[1]-0.05:.2f}, {half[2]-0.05:.2f}])")
        print(f"\n  Suggested start_pose offset from TARGET:")
        print(f"    start = TARGET + [0, {center[1]-TARGET[1]:+.2f}, {center[2]-TARGET[2]:+.2f}]")
    else:
        print("  Nothing reachable. Check move_group and arm_control_node are running.")


if __name__ == "__main__":
    main()
