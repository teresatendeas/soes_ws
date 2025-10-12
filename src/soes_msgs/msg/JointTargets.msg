# Desired joint targets for 4-DOF arm
std_msgs/Header header
float32[4] position       # rad
float32[4] velocity       # rad/s
bool use_velocity         # if true, ESP uses velocity feedforward
