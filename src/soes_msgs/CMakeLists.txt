cmake_minimum_required(VERSION 3.5)
project(soes_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  "msg/BoolStamped.msg"
  "msg/JointTargets.msg"
  "msg/JointFeedback.msg"
  "msg/PumpCmd.msg"
  "msg/RollerCmd.msg"
  "msg/CupcakeCenters.msg"
  "msg/VisionQuality.msg"
  "msg/SpiralParams.msg"
)

set(srv_files
  "srv/StartStop.srv"
  "srv/Home.srv"
  "srv/SetSpiralParams.srv"
  "srv/RollTray.srv"
  "srv/PumpPrime.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
