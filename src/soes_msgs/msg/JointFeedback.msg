# Feedback from ESP for 4-DOF arm
std_msgs/Header header
float32[4] position       # rad
float32[4] velocity       # rad/s
bool at_target
uint8 fault_code          # 0=OK, nonzero=fault
