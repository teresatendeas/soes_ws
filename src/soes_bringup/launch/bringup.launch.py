from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('soes_bringup')
    cfg = os.path.join(pkg_share, 'config')

    return LaunchDescription([
        Node(
            package='soes_state',
            executable='state_node',
            name='soes_state',
            output='screen',
            parameters=[os.path.join(cfg, 'state.yaml')],
        ),
        Node(
            package='soes_robothand',
            executable='robothand_node',
            name='soes_robothand',
            output='screen',
            parameters=[os.path.join(cfg, 'robothand.yaml')],
        ),
        Node(
            package='soes_vision',
            executable='vision_node',
            name='soes_vision',
            output='screen',
            parameters=[os.path.join(cfg, 'vision.yaml')],
        ),
        Node(
            package='soes_state',                      # <— same package
            executable='i2c_bridge',                   # <— new entry point
            name='soes_comm_i2c',                      # <— matches comm.yaml root
            output='screen',
            parameters=[os.path.join(cfg, 'comm.yaml')],
        ),
    ])
