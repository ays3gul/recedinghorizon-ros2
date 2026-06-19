#!/bin/bash
# move_bunny_to.sh — move the bunny to a new (x,y) SYNCHRONOUSLY across
# Gazebo world (both src + install copies) AND code (GT translation + target),
# so reconstruction, GT mesh, and the physical object all stay aligned.
#
# Usage:  ./move_bunny_to.sh <new_x> <new_y>
# Example: ./move_bunny_to.sh 0.5 -0.25
#
# Z is kept at 1.0 (world) / 1.0-z_corr (GT). Only X,Y move.
# After running this you MUST restart the Gazebo stack so the new world loads.
set -e

NEWX="${1:?usage: move_bunny_to.sh <x> <y>}"
NEWY="${2:?usage: move_bunny_to.sh <x> <y>}"
echo "[move] target bunny -> x=$NEWX y=$NEWY (z unchanged)"

STAMP=$(date +%Y%m%d_%H%M%S)

# ---- 1) Gazebo world files (src + install) -------------------------------
WORLDS=(
  "$HOME/ros2_ws/src/ur5e_l515_description/worlds/ur5e_world.sdf"
  "$HOME/ros2_ws/install/ur5e_l515_description/share/ur5e_l515_description/worlds/ur5e_world.sdf"
)
for W in "${WORLDS[@]}"; do
  if [ -f "$W" ]; then
    cp "$W" "$W.bak_$STAMP"
    # bunny pose line currently: <pose>0.5 -0.4 1.0 0 0 0</pose>
    # replace the first two numbers (x y), keep z + rpy.
    sed -i -E "s|(<pose>)[-0-9.]+ [-0-9.]+ (1\.0 0 0 0</pose>)|\1$NEWX $NEWY \2|" "$W"
    echo "[move] world updated: $W"
    grep -n "<pose>$NEWX $NEWY" "$W" | head -1 || echo "  WARN: pose line not matched in $W"
  else
    echo "  WARN: world not found: $W"
  fi
done

# ---- 2) Code: GT translation in the 3 active files -----------------------
BASE="$HOME/Desktop/RecedingHorizon/src/viewpoint_planning/src"
VP="$BASE/viewpoint_planners/viewpoint_planning.py"
FC="$BASE/viewpoint_planners/fair_comparison_config.py"
TG="$BASE/test_gradient_node.py"
TB="$BASE/test_baseline_node.py"

for f in "$VP" "$TG" "$TB"; do
  cp "$f" "$f.bak_$STAMP"
  # translation = np.array([0.5, -0.4, 1.0 - z_corr])  ->  new x,y
  sed -i -E "s|(translation = np\.array\(\[)[-0-9.]+, [-0-9.]+(, 1\.0 - z_corr\]\))|\1$NEWX, $NEWY\2|" "$f"
  echo "[move] GT translation updated: $(basename $f)"
done

# ---- 3) Code: target_position (VP line ~50 + FC default) -----------------
cp "$FC" "$FC.bak_$STAMP"
# FC: return np.array([0.5, -0.4, 1.1])  (target Z stays 1.1)
sed -i -E "s|(return np\.array\(\[)[-0-9.]+, [-0-9.]+(, 1\.1\]\))|\1$NEWX, $NEWY\2|" "$FC"
echo "[move] FC target updated"
# VP: self.target_position = np.array([0.5, -0.4, 1.1])
sed -i -E "s|(self\.target_position = np\.array\(\[)[-0-9.]+, [-0-9.]+(, 1\.1\]\))|\1$NEWX, $NEWY\2|" "$VP"
echo "[move] VP target updated"

echo ""
echo "[move] DONE. Backups: *.bak_$STAMP"
echo "[move] Verify code:"
echo "  grep -n 'target_position = np\|translation = np' $VP"
echo "[move] Verify world:"
echo "  grep -n '<pose>' ${WORLDS[0]} | grep -i bunny -A0 || grep -n '$NEWX $NEWY' ${WORLDS[0]}"
echo ""
echo "[move] *** RESTART the Gazebo stack now so the new world loads. ***"
echo "       Then: gz model --model bunny --pose   (should show $NEWX $NEWY 1.0)"
