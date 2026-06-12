#!/bin/bash
# Run any planner under ROS 2 Jazzy with the rh_nbv_ros2 conda environment.
#
# PREREQUISITE — Gz Sim + robot stack must be running in a separate terminal:
#   source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
#   export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
#   ros2 launch abb_l515_moveit_config_ros2 move_group_gz.launch.py
# Wait for "arm_control_node ready — serving move_arm_to_pose" before running.
#
# Usage:
#   ./run_ros2_gz.sh                              # RH-NBV, defaults
#   PLANNER=gradient OCC=well ./run_ros2_gz.sh    # GradientNBV
#   PLANNER=pso OCC=frontal ./run_ros2_gz.sh      # PSO
#   PLANNER=random OCC=tunnel ./run_ros2_gz.sh    # Random
#   RH_K=20 RH_H=2 NUM_TRIALS=4 ./run_ros2_gz.sh
#   OCC=frontal EXPERIMENT=C ./run_ros2_gz.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="$HOME/ros2_ws"
CONDA_PYTHON="$HOME/miniconda3/envs/rh_nbv_ros2/bin/python"

# Select script based on PLANNER env var (default: rh)
PLANNER="${PLANNER:-rh}"
case "$PLANNER" in
    rh|RH)
        PYTHON_SCRIPT="$SCRIPT_DIR/src/viewpoint_planning/src/test_rh_node.py"
        ;;
    gradient|GradientNBV|gradientnbv)
        PYTHON_SCRIPT="$SCRIPT_DIR/src/viewpoint_planning/src/test_gradient_node.py"
        ;;
    pso|PSO|random|Random)
        PYTHON_SCRIPT="$SCRIPT_DIR/src/viewpoint_planning/src/test_baseline_node.py"
        export PLANNER="$PLANNER"
        ;;
    *)
        echo "[ERROR] Unknown PLANNER='$PLANNER'. Use: rh, gradient, pso, random"
        exit 1
        ;;
esac

# Clear any stale ROS1/Humble environment so Jazzy starts clean
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset ROS_PACKAGE_PATH PYTHONPATH LD_LIBRARY_PATH

# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Source the abb_interfaces colcon workspace
if [ -f "$ROS2_WS/install/setup.bash" ]; then
    source "$ROS2_WS/install/setup.bash"
else
    echo "[ERROR] $ROS2_WS/install/setup.bash not found."
    echo "        Build the workspace first:"
    echo "        cd $ROS2_WS && colcon build"
    exit 1
fi

# Python path: project source trees + ROS 2 Jazzy packages
export PYTHONPATH=\
$SCRIPT_DIR/src/viewpoint_planning/src:\
$SCRIPT_DIR/src/robot/abb_control/src:\
$SCRIPT_DIR/src/common/utils/src:\
/opt/ros/jazzy/lib/python3.12/site-packages:\
$ROS2_WS/install/abb_interfaces/lib/python3.12/site-packages

export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:/opt/ros/jazzy/opt/gz_transport_vendor/lib:$ROS2_WS/install/abb_interfaces/lib:$ROS2_WS/install/abb_control/lib

export PYTHONUNBUFFERED=1

# Tell rclpy to use Gz simulation clock
export USE_SIM_TIME=true

echo "[run_ros2_gz.sh] Planner: $PLANNER | Script: $(basename $PYTHON_SCRIPT)"

exec "$CONDA_PYTHON" -u \
    "$PYTHON_SCRIPT" \
    --ros-args -p use_sim_time:=true "$@"
