# Conveyor/roller motion command
std_msgs/Header header
float32 speed_mm_s        # signed; + away from arm
float32 distance_mm       # 0=continuous; else move this distance
bool relative             # true=relative move
