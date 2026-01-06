# 🦾 SOES (Smart Optimized Extrusion System)

A **4-DOF robot arm** for automated pastry extrusion (“kue soes”) powered by **ROS 2 Foxy** on a **Jetson Nano**.  
The workspace contains all core components: vision detection, inverse kinematics, custom ROS messages, and the main state machine that controls the entire extrusion process.

---

## ⚙️ Packages Summary

| Package | Description |
|----------|--------------|
| **soes_bringup** | Contains launch files (`bringup.launch.py`) and YAML configs to start all nodes together. |
| **soes_state** | Core **state machine** controlling system logic, generating and moving **spiral trajectories** for the 4-DOF arm. |
| **soes_vision** | Handles **camera detection** — publishes simple `True/False` for tray or dough presence. |
| **soes_msgs** | Defines custom messages/services if you later need them (e.g., `Angles.msg`, `StartStop.srv`). |


## ⚙️ Quick Start
```bash
# Source ROS 2 Foxy
source /opt/ros/foxy/setup.bash

# Build and source workspace
cd ~/soes_ws
colcon build
source install/setup.bash

# Launch the full system
ros2 launch soes_bringup bringup.launch.py




