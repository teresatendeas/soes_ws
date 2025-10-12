from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    nodes = []

    # --- State Machine Node ---
    nodes.append(
        Node(
            package='soes_state',
            executable='state_node',
            name='soes_state',
            output='screen',
            parameters=['/home/teresatendeas/soes_ws/src/soes_bringup/config/state.yaml'],
        )
    )

    # --- Robot Hand Node (optional) ---
    nodes.append(
        Node(
            package='soes_robothand',
            executable='robothand_node',
            name='soes_robothand',
            output='screen',
            parameters=['/home/teresatendeas/soes_ws/src/soes_bringup/config/robothand.yaml'],
            condition=None,
        )
    )

    # --- Vision Node (optional) ---
    nodes.append(
        Node(
            package='soes_vision',
            executable='vision_node',
            name='soes_vision',
            output='screen',
            parameters=['/home/teresatendeas/soes_ws/src/soes_bringup/config/vision.yaml'],
            condition=None,
        )
    )

    return LaunchDescription(nodes)
