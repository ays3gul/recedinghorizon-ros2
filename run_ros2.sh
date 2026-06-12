#!/bin/bash
# Run test_rh_node.py under ROS 2 Jazzy with the rh_nbv_ros2 conda environment.
#
# PREREQUISITE — robot stack must be running in a separate terminal:
#   source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
#   ros2 launch abb_l515_moveit_config_ros2 move_group.launch.py
# Wait for "arm_control_node ready — serving move_arm_to_pose" before running this script.
#
# Usage:
#   ./run_ros2.sh                        # defaults (K=10 H=3, 1 trial)
#   RH_K=20 RH_H=2 NUM_TRIALS=4 ./run_ros2.sh
#   OCC=frontal EXPERIMENT=C ./run_ros2.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="$HOME/ros2_ws"
CONDA_PYTHON="$HOME/miniconda3/envs/rh_nbv_ros2/bin/python"

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
    echo "        cd $ROS2_WS && colcon build --cmake-args \\"
    echo "          -DPython3_EXECUTABLE=/usr/bin/python3.12 \\"
    echo "          \"-DPython3_NumPy_INCLUDE_DIR=\$(/usr/bin/python3.12 -c 'import numpy; print(numpy.get_include())')\""
    exit 1
fi

# Python path: project source trees + ROS 2 Jazzy packages
export PYTHONPATH=\
$SCRIPT_DIR/src/viewpoint_planning/src:\
$SCRIPT_DIR/src/robot/abb_control/src:\
$SCRIPT_DIR/src/common/utils/src:\
/opt/ros/jazzy/lib/python3.12/site-packages:\
$ROS2_WS/install/abb_interfaces/lib/python3.12/site-packages

export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$ROS2_WS/install/abb_interfaces/lib:$ROS2_WS/install/abb_control/lib
export PYTHONUNBUFFERED=1

exec "$CONDA_PYTHON" -u \
    "$SCRIPT_DIR/src/viewpoint_planning/src/test_rh_node.py" "$@"
