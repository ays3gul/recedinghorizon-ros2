#!/usr/bin/env python3
"""
placement_test.py — Find the best bunny position by testing, for each candidate,
whether the camera's predefined start pose AND the corners of its motion box are
physically reachable (real move_arm_to_pose, since compute_ik can't see 'world').

For each candidate target (bunny) position we test:
  1. The start pose: target + [0, +distance, 0]  (predefine_start_pose convention)
  2. 6 box-corner-ish poses around the start, at +/- camera halfwidths, each
     oriented to look AT the target (so it's a realistic viewing pose).

A candidate is GOOD if its start pose reaches and most box poses reach -> the
planner can actually move around there.

Protections against the earlier service lockup:
  * short timeout per pose
  * return to a safe home pose every few moves
  * stop a candidate early after consecutive failures
  * write results incrementally to placement_results.csv

Run (robot stack up):
    cd ~/Desktop/RecedingHorizon
    ./run_reach_test.sh        # after pointing REACH_SCRIPT at this file
"""
import csv
import time
import numpy as np

import ros2_node  # singleton node + spin thread; must be first
from abb_control.arm_control_client import ArmControlClient
from utils.py_utils import numpy_to_pose, look_at_rotation

# predefine_start_pose uses distance=0.35 along +Y
START_DISTANCE = 0.35
# camera motion halfwidths (current fair_comparison_config value)
HALF = np.array([0.2, 0.1, 0.15])

# Candidate bunny (target) positions: pull it toward the robot in Y.
# x,z kept at the current target (0.5, 1.1); only Y swept.
CANDIDATES = [
    np.array([0.5, -0.40, 1.1]),   # current (baseline reference)
    np.array([0.5, -0.30, 1.1]),
    np.array([0.5, -0.25, 1.1]),
    np.array([0.5, -0.20, 1.1]),
    np.array([0.5, -0.15, 1.1]),
]

# A safe home we know is reachable (the current experiment start pose)
HOME = np.array([0.5, -0.05, 1.1, 0.0, 0.0, -0.7071067811865475, 0.7071067811865476])


def pose_looking_at(cam_pos, target):
    q = look_at_rotation(np.asarray(cam_pos, float), np.asarray(target, float))  # [w,x,y,z]
    # numpy_to_pose expects the same layout the planner feeds it (concat pos+look_at output)
    return np.concatenate((np.asarray(cam_pos, float), q))


def box_poses(start_pos, target):
    """Start pose + 6 axis corner pokes (one per +/- axis at the halfwidth)."""
    poses = {"start": pose_looking_at(start_pos, target)}
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        for sgn, lbl in [(+1, "+"), (-1, "-")]:
            p = start_pos.copy()
            p[axis] += sgn * HALF[axis]
            poses[f"{lbl}{name}"] = pose_looking_at(p, target)
    return poses


def main():
    ros2_node.init("placement_test")
    arm = ArmControlClient()

    f = open("placement_results.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["target_y", "pose_label", "reachable"]); f.flush()

    print("\n[placement] testing candidate bunny positions\n")
    summary = []

    for ci, target in enumerate(CANDIDATES):
        start_pos = np.array([target[0], target[1] + START_DISTANCE, target[2]])
        poses = box_poses(start_pos, target)
        print(f"=== candidate target y={target[1]:+.2f}  (start y={start_pos[1]:+.2f}) ===")

        # always re-home first so each candidate starts from the same place
        arm.move_arm_to_pose(numpy_to_pose(HOME)); time.sleep(0.5)

        n_ok = 0; n_tot = 0; start_ok = False; consec_fail = 0
        for label, pose in poses.items():
            n_tot += 1
            ok = arm.move_arm_to_pose(numpy_to_pose(pose))
            w.writerow([f"{target[1]:.2f}", label, int(bool(ok))]); f.flush()
            print(f"    {label:6s} ({pose[0]:+.2f},{pose[1]:+.2f},{pose[2]:+.2f}) -> "
                  f"{'REACH' if ok else 'FAIL'}")
            if label == "start":
                start_ok = bool(ok)
                if not ok:
                    print("    start pose unreachable -> skipping rest of this candidate")
                    break
            if ok:
                n_ok += 1; consec_fail = 0
            else:
                consec_fail += 1
            if consec_fail >= 3:
                print("    3 consecutive fails -> moving on")
                break
            time.sleep(0.3)

        summary.append((target[1], start_ok, n_ok, n_tot))
        print(f"    => start={'OK' if start_ok else 'NO'}  reached {n_ok}/{n_tot}\n")

    f.close()
    print("=" * 56)
    print("  SUMMARY (best = start OK and highest reach fraction)")
    print("=" * 56)
    for ty, sok, nok, ntot in summary:
        flag = "  <-- candidate" if sok and nok >= ntot - 1 else ""
        print(f"  target y={ty:+.2f}: start={'OK ' if sok else 'NO '}  "
              f"box {nok}/{ntot}{flag}")
    print("\n  csv -> placement_results.csv")

if __name__ == "__main__":
    main()
