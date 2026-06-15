# RH-NBV: UR5e + RealSense D455 — ROS 2 Jazzy

## Repositories
- recedinghorizon-ros2 — planner + experiment scripts
- ur5e-l515-ros2 — ROS 2 packages

## Installation (fresh machine)

### 1. ROS 2 apt dependencies
sudo apt update
sudo apt install -y ros-jazzy-ur ros-jazzy-ur-simulation-gz ros-jazzy-ur-moveit-config ros-jazzy-realsense2-description ros-jazzy-ros-gz ros-jazzy-moveit

### 2. Clone repositories
git clone https://github.com/ays3gul/ur5e-l515-ros2.git ~/ros2_ws/src
git clone https://github.com/ays3gul/recedinghorizon-ros2.git ~/Desktop/RecedingHorizon

### 3. Build the colcon workspace
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash

### 4. Create the conda environment
conda env create -f ~/Desktop/RecedingHorizon/environment.yml
conda activate rh_nbv_ros2

## Running

### Terminal 1 — simulation + robot stack
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$HOME/ros2_ws/install/ur5e_l515_description/share
ros2 launch ur5e_l515_description move_group_gz_ur5e.launch.py

### Terminal 2 — planner (wait for "arm_control_node ready" first)
cd ~/Desktop/RecedingHorizon
PLANNER=pso OCC=none EXPERIMENT=D ./run_ros2_gz.sh

## Options
- PLANNER: rh, gradient, pso, random
- OCC: none, frontal, half_box, tunnel, well
