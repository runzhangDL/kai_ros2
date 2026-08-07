"""Bring up just the calibration status node.

    ros2 launch humanoid_calibration calibration_status.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "period_s",
                default_value="5.0",
                description="How often the calibration file is re-validated.",
            ),
            Node(
                package="humanoid_calibration",
                executable="calibration_status",
                name="calibration_status",
                output="screen",
                parameters=[{"period_s": LaunchConfiguration("period_s")}],
            ),
        ]
    )
