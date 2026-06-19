#!/usr/bin/env python3
"""
reach_map_multi.py — Test the reachable orbit ARC around SEVERAL candidate
bunny positions in a single run, WITHOUT moving the bunny in Gazebo.

Why this works: arm reachability is a physical fact about whether the UR5e
can place its end-effector at a given (position, orientation). It does not
depend on where the bunny actually is in the world. So we can ask, for each
candidate center C, "if the bunny were at C, how far around it could the
camera orbit?" by sweeping camera poses on a circle around C and looking
toward C. The winner is the candidate with the widest reachable arc.

We move the bunny in Gazebo only ONCE, after we pick the best candidate.

Azimuth: 0=front(+Y toward robot side), 90=+X(right), 180=behind(-Y),
270=-X(left). Camera looks at the candidate center each time.

Run (robot stack up), via run_reach_test.sh pointing at this file (copied
over reach_test.py), same as before.
"""
import csv
import time
import numpy as np

import ros2_node  # singleton node + spin thread; must be first
from abb_control.arm_control_client import ArmControlClient
from utils.py_utils import numpy_to_pose, look_at_rotation

# Robot base is at origin. Current bunny is [0.5,-0.4,1.0].
# Candidate centers pull the bunny toward the robot (smaller |y|, maybe smaller x).
CANDIDATES = {
    "cur_050_-040": np.array([0.50, -0.40, 1.0]),   # current (baseline ref)
    "y_050_-030":   np.array([0.50, -0.30, 1.0]),
    "y_050_-020":   np.array([0.50, -0.20, 1.0]),
    "xy_045_-020":  np.array([0.45, -0.20, 1.0]),
    "xy_045_-025":  np.array([0.45, -0.25, 1.0]),
}

# Orbit radius around the candidate center. Smaller = camera closer, easier
# to reach around. We test one moderate radius; can rerun with others.
RADIUS = 0.25
Z_OFF = 0.05

# Azimuth samples (deg). 12 steps = every 30 deg (faster than 16).
N_AZ = 12
AZIMUTHS = np.linspace(0.0, 360.0, N_AZ, endpoint=False)

# Safe home = current experiment start pose (known reachable)
HOME = np.array([0.5, -0.05, 1.0, 0.0, 0.0,
                 -0.7071067811865475, 0.7071067811865476])

# Per-pose timeout guard: after this many consecutive fails, re-home and
# move to the next candidate (avoids the 180s lockups seen before).
MAX_CONSEC_FAIL = 4


def cam_on_circle(center, radius, az_deg, z_off):
    az = np.radians(az_deg)
    x = center[0] + radius * np.sin(az)   # az=0->+Y, az=90->+X
    y = center[1] + radius * np.cos(az)
    z = center[2] + z_off
    return np.array([x, y, z])


def pose_looking_at(cam_pos, target):
    q = look_at_rotation(np.asarray(cam_pos, float), np.asarray(target, float))
    return np.concatenate((np.asarray(cam_pos, float), q))


def main():
    ros2_node.init("reach_map_multi")
    arm = ArmControlClient()

    f = open("reach_map_multi_results.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["candidate", "cx", "cy", "az_deg", "x", "y", "z", "reachable"])
    f.flush()

    print(f"\n[reach_map_multi] radius={RADIUS} z_off=+{Z_OFF}  azimuth step={360/N_AZ:.0f}deg\n")

    per_cand = {}
    for name, C in CANDIDATES.items():
        print(f"\n=== candidate {name}  center={C} ===")
        arm.move_arm_to_pose(numpy_to_pose(HOME)); time.sleep(0.5)

        reached = []
        consec = 0
        skipped = False
        for az in AZIMUTHS:
            if skipped:
                break
            cam = cam_on_circle(C, RADIUS, az, Z_OFF)
            pose = pose_looking_at(cam, C)
            ok = arm.move_arm_to_pose(numpy_to_pose(pose))
            w.writerow([name, f"{C[0]:.2f}", f"{C[1]:.2f}", f"{az:.0f}",
                        f"{cam[0]:.3f}", f"{cam[1]:.3f}", f"{cam[2]:.3f}",
                        int(bool(ok))]); f.flush()
            print(f"    az={az:5.0f} ({cam[0]:+.2f},{cam[1]:+.2f},{cam[2]:+.2f}) -> "
                  f"{'REACH' if ok else 'FAIL'}")
            if ok:
                reached.append(az); consec = 0
            else:
                consec += 1
            if consec >= MAX_CONSEC_FAIL:
                print(f"    {MAX_CONSEC_FAIL} consecutive fails -> re-home, next candidate")
                arm.move_arm_to_pose(numpy_to_pose(HOME)); time.sleep(0.5)
                skipped = True
            time.sleep(0.2)

        per_cand[name] = (C, reached)

    f.close()

    # ---- Summary -------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"  REACHABLE ORBIT ARC per candidate  (radius={RADIUS})")
    print("  az: 0=front 90=right 180=behind 270=left")
    print("=" * 64)
    best_name, best_n = None, -1
    for name, (C, azs) in per_cand.items():
        n = len(azs)
        arc = (f"{n}/{N_AZ} poses  az=[{','.join(f'{a:.0f}' for a in azs)}]"
               if n else "NONE")
        print(f"  {name:<16} center=({C[0]:.2f},{C[1]:.2f}): {arc}")
        if n > best_n:
            best_name, best_n = name, n
    print("=" * 64)
    print(f"\n  WIDEST ARC: {best_name}  ({best_n}/{N_AZ} reachable)")
    print(f"  -> move the bunny here, then size the reach box to this arc.")
    print(f"  csv -> reach_map_multi_results.csv")


if __name__ == "__main__":
    main()
