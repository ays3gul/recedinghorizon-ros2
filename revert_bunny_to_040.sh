#!/bin/bash
# revert_bunny_to_040.sh — undo ONLY the bunny y-move (-0.25 -> -0.40),
# keeping the reach-bounds fix in place. Bunny stays at -0.40 in Gazebo
# (ur5e_world.sdf), so this re-aligns code with the actual sim object.
#
# Rationale: the coverage gain (39%->63%) came from the reach fix, not from
# moving the bunny. So we keep the object where the sim has it (-0.40) and
# keep the reach fix. Result: GT, target, and Gazebo all agree at -0.40,
# F1 returns, coverage stays high, and no occlusion panels need moving.
set -e
BASE="$HOME/Desktop/RecedingHorizon/src/viewpoint_planning/src"
VP="$BASE/viewpoint_planners/viewpoint_planning.py"
FC="$BASE/viewpoint_planners/fair_comparison_config.py"
TG="$BASE/test_gradient_node.py"
TB="$BASE/test_baseline_node.py"

STAMP=$(date +%Y%m%d_%H%M%S)
echo "[revert] backing up to *.bak_revert_$STAMP"
for f in "$VP" "$FC" "$TG" "$TB"; do cp "$f" "$f.bak_revert_$STAMP"; done

# 1) FC target default back to -0.4
sed -i "s|return np.array(\[0.5, -0.25, 1.1\])|return np.array([0.5, -0.4, 1.1])|" "$FC"
echo "[revert] FC target -> 0.5, -0.4, 1.1"

# 2) GT translation back to -0.4 in the 3 active files
for f in "$VP" "$TG" "$TB"; do
    sed -i "s|translation = np.array(\[0.5, -0.25, 1.0 - z_corr\])|translation = np.array([0.5, -0.4, 1.0 - z_corr])|" "$f"
    echo "[revert] GT translation -> -0.4 in $(basename $f)"
done

# 3) VP target default back to -0.4
sed -i "s|self.target_position = np.array(\[0.5, -0.25, 1.1\])|self.target_position = np.array([0.5, -0.4, 1.1])|" "$VP"
echo "[revert] VP target -> 0.5, -0.4, 1.1"

# NOTE: reach_bounds_for_start fix is intentionally LEFT IN PLACE.
echo ""
echo "[revert] DONE (reach fix kept). Verify:"
echo "  grep -n 'target_position\|translation = np\|reach_bounds_for_start' $VP"
