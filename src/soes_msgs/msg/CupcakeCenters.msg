# Pixel or metric centers for the 3 cupcakes (left, center, right)
std_msgs/Header header
geometry_msgs/Point[] centers   # length=3; x,y used; z optional
string frame_id                 # e.g., "camera" or "robot_base"
