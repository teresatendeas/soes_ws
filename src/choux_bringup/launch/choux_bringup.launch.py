from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('choux_bringup'), 'config', 'extruder.yaml'
    )
    return LaunchDescription([
        Node(
            package='choux_control',
            executable='extruder_node',
            name='extruder_node',
            output='screen',
            parameters=[cfg],
        )
    ])
