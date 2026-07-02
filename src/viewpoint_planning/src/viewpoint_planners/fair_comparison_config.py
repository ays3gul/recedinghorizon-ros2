"""
fair_comparison_config.py — single source of truth for every parameter that
MUST be identical between the RH-NBV planner and the GradientNBV baseline.

Both test_rh_node.py (via viewpoint_planning.py) and test_gradient_node.py
import from here, so the two planners can never silently drift apart again.
Any value that affects the reconstruction volume, the evaluation ROI, the
camera workspace, the robot reach, or the per-trial start perturbation lives
HERE and nowhere else.

Rationale for each value is documented inline; all are overridable via env
vars so ablations stay reproducible.
"""
import os
import numpy as np

# ---------------------------------------------------------------------------
# Target node (centre of the ROI on the object surface).
# Camera starts at Y=+0.20 (target_y + 0.60 m standoff) and looks toward -Y.
# ---------------------------------------------------------------------------
def get_target_position() -> np.ndarray:
    tp_env = os.environ.get("TARGET_POS")
    if tp_env:
        return np.array([float(v) for v in tp_env.split(",")])
    target = os.environ.get("TARGET", "bunny").lower()
    if target == "tomato":
        # Fruit1-4 centroid in world space (pose=0.5 -0.50 0.9, scale=0.4)
        return np.array([0.50, -0.50, 1.16])
    # Bunny spawn (0.50, -0.30, 1.0). COLLADA node matrix gives world Z∈[0.980,1.165],
    # geometric centre Z=1.073. ROI z=1.073±0.075=[0.998,1.148] covers 97% of bunny height.
    # Camera starts at y=-0.30+0.60=+0.30. Workspace y∈[+0.20,+0.40].
    # Camera-bunny y∈[0.50,0.70m] = D455 ideal range.
    return np.array([0.50, -0.30, 1.07])



# ---------------------------------------------------------------------------
# VoxelGrid reconstruction volume.
# The depth axis is Y (camera looks along -Y), so the grid is widest in Y.
# This MUST be identical for both planners — a different grid_size changes the
# physical reconstruction volume, the coverage denominator, and which voxels
# survive the ROI crop, invalidating any RH-vs-GradientNBV comparison.
# ---------------------------------------------------------------------------
GRID_SIZE  = np.array([0.3, 0.6, 0.3])
VOXEL_SIZE = np.array([0.003])


# ---------------------------------------------------------------------------
# Evaluation ROI and F1 matching threshold.
# Mirrors voxel_grid.set_target_roi (+/-25 voxels = +/-75mm at 3mm voxels).
# F1_THRESH default = 4x voxel size (~12mm) to account for the reconstructed
# occupancy shell sitting ~9-13mm off the continuous mesh surface (true for
# BOTH planners in this setup). Identical for both => fair.
# ---------------------------------------------------------------------------
ROI_HALF        = float(os.environ.get("ROI_HALF", 0.095))  # ±31 voxels @ 3mm = 93mm ≈ bunny z_max
F1_THRESH_SCALE = 4.0  # x voxel_size


# ---------------------------------------------------------------------------
# Axis-aligned camera workspace: start_pose +/- this half-width.
# This is Burusa's GradientNBV box; RH mirrors it exactly in rh_planner.py.
# ---------------------------------------------------------------------------
# UR5e + D455. Bunny at y=-0.30, standoff 0.60 m → start_y=+0.30. With ±0.10
# bounds in y, camera-bunny stays in [0.50, 0.70 m] = D455 ideal range exactly.
# x_hw=0.28: camera reaches x=[0.22,0.78], 25° side-view angle.
# z_hw=0.22: camera reaches z=[0.85,1.29], top-down views of bunny back/ears.
CAMERA_BOUNDS_HALFWIDTHS = np.array([0.28, 0.10, 0.22], dtype=np.float32)


# ---------------------------------------------------------------------------
# RH's robot reach clamp. RH keeps its own internal reach-clamp mechanism
# (needed on the real arm), but for the fair simulation comparison its bounds
# are set EQUAL to the shared camera_bounds box, so RH's clamp can never make
# its usable workspace smaller than GradientNBV's. Both planners then search
# the identical physical region; only their search STRATEGY differs.
#
# Because camera_bounds depend on the (jittered) start pose, the reach box is
# computed per start pose via reach_bounds_for_start(). The old fixed box
# [[0.30,-0.15,0.97],[0.65,0.05,1.25]] left RH only 55-82% of GradientNBV's
# volume — an unfair handicap — and is replaced by this start-aligned box.
# ---------------------------------------------------------------------------
def reach_bounds_for_start(start_pose: np.ndarray) -> np.ndarray:
    """Reach box = camera_bounds box, i.e. start_pos +/- CAMERA_BOUNDS_HALFWIDTHS.
    Returns shape (2,3): [lo, hi]. Identical to the box GradientNBV clamps to,
    so RH and GradientNBV share the exact same usable camera workspace."""
    s = np.asarray(start_pose, dtype=np.float32)[:3]
    return np.array([s - CAMERA_BOUNDS_HALFWIDTHS,
                     s + CAMERA_BOUNDS_HALFWIDTHS], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-trial start-pose perturbation (Burusa's +/-3cm ROI/start uncertainty).
# Trial 0 is deterministic (seed None); trial t>0 uses BASE_SEED + t.
# Both planners MUST use this same logic so trial t starts both from the same
# perturbed pose (paired comparison + matched variance).
# ---------------------------------------------------------------------------
START_JITTER = 0.03
BASE_SEED    = int(os.environ.get("BASE_SEED", 42))


def seed_for_trial(trial_idx: int):
    """Deterministic trial 0; reproducible jitter seed for later trials."""
    return None if trial_idx == 0 else (BASE_SEED + trial_idx)


def jitter_start_pose(start_pose: np.ndarray, trial_idx: int) -> np.ndarray:
    """Apply the same reproducible +/-START_JITTER perturbation both planners
    use. Operates on the position (first 3 entries) only; orientation is
    recomputed by look_at toward the target downstream. Returns a copy."""
    pose = np.array(start_pose, dtype=float).copy()
    seed = seed_for_trial(trial_idx)
    if seed is not None:
        rng = np.random.default_rng(seed)
        pose[:3] = pose[:3] + rng.uniform(-START_JITTER, START_JITTER, size=3)
    return pose
