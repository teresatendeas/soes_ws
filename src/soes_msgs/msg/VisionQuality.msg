# Icing size/quality assessment for 3 cupcakes
std_msgs/Header header
float32[] diameter_mm     # length=3
float32[] score           # 0..1 confidence per cupcake (length=3)
bool needs_human          # true => turn on LED / notify operator
