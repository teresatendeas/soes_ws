# 🦾 SOES (Smart Optimized Extrusion System)

A **4-DOF robot arm** for automated pastry extrusion (“kue soes”) powered by **ROS 2 Foxy** on a **Jetson Nano**.  
The workspace contains all core components: vision detection, inverse kinematics, custom ROS messages, and the main state machine that controls the entire extrusion process.

---

## 🧱 Workspace Layout
soes_ws/
├─ src/
│ ├─ soes_msgs/ → custom ROS 2 message & service definitions
│ ├─ soes_bringup/ → launch & parameter configs
│ ├─ soes_state/ → main system logic & state machine
│ ├─ soes_robothand/ → spiral generator + inverse kinematics
│ └─ soes_vision/ → camera detection node

yaml
Copy code

---

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
