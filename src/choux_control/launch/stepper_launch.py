from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='choux_control',
            executable='stepper_node',
            name='stepper_node',
            parameters=[{'rpm': 120}]
        )
    ])
