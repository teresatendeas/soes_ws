# Conveyor/roller motion command
std_msgs/Header header
bool on          # True = conveyor ON, False = OFF
float32 duty     # optional, biar sama seperti PumpCmd (0.0–1.0)
float32 duration_s  # optional, 0.0 = continuous
