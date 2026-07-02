#!/bin/bash
# Run reach_test.py in the SAME environment as run_ros2_gz.sh.
#
# PREREQUISITE (same as the experiments): the Gz Sim + robot stack must be
# running in a separate terminal:
#   source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
#   export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/jazzy/lib
#   ros2 launch ur5e_l515_description move_group_gz_ur5e.launch.py
# Wait until arm_control is ready, then run this script.
#
# Usage:
#   ./run_reach_test.sh                 # runs reach_test_ur5e.py (default)
#   REACH=reach_map.py ./run_reach_test.sh   # run a different probe script
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="$HOME/ros2_ws"
CONDA_PYTHON="$HOME/miniconda3/envs/rh_nbv_ros2/bin/python"

# Probe script next to this script; defaults to the UR5e reach test.
REACH_SCRIPT="$SCRIPT_DIR/${REACH:-reach_test_ur5e.py}"
if [ ! -f "$REACH_SCRIPT" ]; then
    echo "[ERROR] $REACH_SCRIPT not found in $SCRIPT_DIR"
    exit 1
fi

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
    exit 1
fi

# Python path: project source trees + ROS 2 Jazzy packages (same as run_ros2_gz.sh)
export PYTHONPATH=\
$SCRIPT_DIR/src/viewpoint_planning/src:\
$SCRIPT_DIR/src/robot/abb_control/src:\
$SCRIPT_DIR/src/common/utils/src:\
/opt/ros/jazzy/lib/python3.12/site-packages:\
$ROS2_WS/install/abb_interfaces/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:/opt/ros/jazzy/opt/gz_transport_vendor/lib:$ROS2_WS/install/abb_interfaces/lib:$ROS2_WS/install/abb_control/lib
export PYTHONUNBUFFERED=1
export USE_SIM_TIME=true

echo "[run_reach_test.sh] Running $(basename "$REACH_SCRIPT") in rh_nbv_ros2 env"
exec "$CONDA_PYTHON" -u "$REACH_SCRIPT" --ros-args -p use_sim_time:=true "$@"
