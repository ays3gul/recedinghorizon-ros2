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

# Real UR5e (0.85 m reach, base at world origin on a 0.525 m pedestal) + real
# D455 (min 0.40 m, ideal 0.60 m), eye-in-hand, scanning a real object (e.g. a
# coffee mug ~0.1 m) on the table. Goal: find the (position, standoff) where the
# arm can orbit the object the most, at a D455-VALID standoff — so a mug placed
# there causes no reachability/blind-zone problems.
#
# Candidates span table positions and heights (z = mug-center height). Lower z is
# nearer the arm's comfortable working height (shoulder ~0.69 m).
CANDIDATES = {
    "x050_y-030_z100": np.array([0.50, -0.30, 1.00]),  # current table height
    "x050_y-030_z090": np.array([0.50, -0.30, 0.90]),
    "x050_y-030_z085": np.array([0.50, -0.30, 0.85]),
    "x055_y-035_z095": np.array([0.55, -0.35, 0.95]),
    "x045_y-025_z090": np.array([0.45, -0.25, 0.90]),
    "x060_y-030_z090": np.array([0.60, -0.30, 0.90]),
}

# D455-VALID orbit radii (standoff): 0.40 = min, 0.50 = comfortable, 0.60 = ideal.
RADII = [0.40, 0.50, 0.60]
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
    w.writerow(["candidate", "cx", "cy", "cz", "radius", "az_deg", "x", "y", "z", "reachable"])
    f.flush()

    print(f"\n[reach_map_multi] radii={RADII} z_off=+{Z_OFF}  azimuth step={360/N_AZ:.0f}deg\n")

    per_combo = {}   # (name, radius) -> (C, reached_azimuths)
    for name, C in CANDIDATES.items():
        for radius in RADII:
            print(f"\n=== {name}  center={C}  radius={radius:.2f} ===")
            arm.move_arm_to_pose(numpy_to_pose(HOME)); time.sleep(0.5)

            reached = []
            consec = 0
            skipped = False
            for az in AZIMUTHS:
                if skipped:
                    break
                cam = cam_on_circle(C, radius, az, Z_OFF)
                pose = pose_looking_at(cam, C)
                ok = arm.move_arm_to_pose(numpy_to_pose(pose))
                w.writerow([name, f"{C[0]:.2f}", f"{C[1]:.2f}", f"{C[2]:.2f}",
                            f"{radius:.2f}", f"{az:.0f}",
                            f"{cam[0]:.3f}", f"{cam[1]:.3f}", f"{cam[2]:.3f}",
                            int(bool(ok))]); f.flush()
                print(f"    az={az:5.0f} ({cam[0]:+.2f},{cam[1]:+.2f},{cam[2]:+.2f}) -> "
                      f"{'REACH' if ok else 'FAIL'}")
                if ok:
                    reached.append(az); consec = 0
                else:
                    consec += 1
                if consec >= MAX_CONSEC_FAIL:
                    print(f"    {MAX_CONSEC_FAIL} consecutive fails -> re-home, next combo")
                    arm.move_arm_to_pose(numpy_to_pose(HOME)); time.sleep(0.5)
                    skipped = True
                time.sleep(0.2)

            per_combo[(name, radius)] = (C, reached)

    f.close()

    # ---- Summary -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  REACHABLE ORBIT ARC per (position, standoff)")
    print("  az: 0=front(+Y) 90=right(+X) 180=behind(-Y) 270=left(-X)")
    print("=" * 70)
    best_combo, best_n = None, -1
    for (name, radius), (C, azs) in per_combo.items():
        n = len(azs)
        arc = (f"{n}/{N_AZ}  az=[{','.join(f'{a:.0f}' for a in azs)}]" if n else "NONE")
        print(f"  {name:<18} r={radius:.2f}: {arc}")
        if n > best_n:
            best_combo, best_n = (name, radius), n
    print("=" * 70)
    if best_combo:
        (bname, bradius) = best_combo
        bC = CANDIDATES[bname]
        print(f"\n  WIDEST ARC: {bname} @ standoff {bradius:.2f} m  ({best_n}/{N_AZ})")
        print(f"  -> object center {bC.tolist()}, D455-valid standoff {bradius:.2f} m.")
        print(f"     Put the mug here; size the reach box / start distance to this.")
    print(f"  csv -> reach_map_multi_results.csv")


if __name__ == "__main__":
    main()
