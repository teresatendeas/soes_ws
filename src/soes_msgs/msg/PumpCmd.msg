# Command to icing pump
std_msgs/Header header
bool on                   # true=enable pump
float32 duty              # 0..1 (optional; 0 if unused)
float32 duration_s        # 0=continuous, >0 run for this long
