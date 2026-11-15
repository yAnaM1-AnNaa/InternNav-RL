"""Start the AgileX base, InternNav velocity bridge and shared camera publisher."""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    here = Path(__file__).resolve().parent
    args = [
        DeclareLaunchArgument("can_interface", default_value="can0"),
        DeclareLaunchArgument("is_bunker_mini", default_value="false"),
        DeclareLaunchArgument("base_odom_topic", default_value="/odom_bridge"),
        DeclareLaunchArgument("odom_frame", default_value="odom"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("max_linear", default_value="0.3", description="Maximum absolute speed (m/s)"),
        DeclareLaunchArgument("max_angular", default_value="0.4", description="Maximum absolute yaw rate (rad/s)"),
        DeclareLaunchArgument("command_timeout", default_value="0.5", description="Seconds without a Twist before stopping"),
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument("camera_python", default_value=sys.executable),
    ]
    base = Node(
        package="bunker_base",
        executable="bunker_base_node",
        parameters=[{
            "port_name": ParameterValue(LaunchConfiguration("can_interface"), value_type=str),
            "is_bunker_mini": ParameterValue(LaunchConfiguration("is_bunker_mini"), value_type=bool),
            "odom_topic_name": ParameterValue(LaunchConfiguration("base_odom_topic"), value_type=str),
            "odom_frame": ParameterValue(LaunchConfiguration("odom_frame"), value_type=str),
            "base_frame": ParameterValue(LaunchConfiguration("base_frame"), value_type=str),
        }],
        remappings=[("/cmd_vel", "/internnav/bunker/cmd_vel")],
        output="screen",
        on_exit=[Shutdown(reason="Bunker base exited")],
    )
    bridge = Node(
        executable=sys.executable,
        arguments=[str(here / "velocity_bridge.py")],
        parameters=[{
            name: ParameterValue(LaunchConfiguration(name), value_type=float)
            for name in ("max_linear", "max_angular", "command_timeout")
        }],
        output="screen",
        on_exit=[Shutdown(reason="Bunker velocity bridge exited")],
    )
    camera = ExecuteProcess(
        cmd=[LaunchConfiguration("camera_python"), str(here.parent / "go2" / "realsense_publisher.py")],
        condition=IfCondition(LaunchConfiguration("start_camera")),
        output="screen",
        on_exit=[Shutdown(reason="RealSense publisher exited")],
    )
    return LaunchDescription([*args, base, bridge, camera])
