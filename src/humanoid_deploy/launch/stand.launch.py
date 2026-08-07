"""Bring up the standing policy on the real robot.

    # dry run first -- nothing is written to a servo
    ros2 launch humanoid_deploy stand.launch.py bundle:=/path/policy_bundle.npz

    # once the dry run looks right
    ros2 launch humanoid_deploy stand.launch.py bundle:=... dry_run:=false

Then, holding the robot upright:
    ros2 service call /humanoid_servo/arm std_srvs/srv/Trigger

The calibration gate runs before any node starts, so an uncalibrated robot
never gets as far as opening the bus.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from humanoid_calibration.launch_guard import calibration_gate


def generate_launch_description():
    share = get_package_share_directory("humanoid_deploy")
    default_config = os.path.join(share, "config", "deploy.yaml")

    bundle = LaunchConfiguration("bundle")
    config = LaunchConfiguration("config")
    dry_run = LaunchConfiguration("dry_run")
    rate = LaunchConfiguration("control_rate_hz")

    return LaunchDescription([
        DeclareLaunchArgument("bundle", description="path to policy_bundle.npz"),
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("dry_run", default_value="true"),
        # 0.0 = use the rate recorded in the bundle (25 Hz, as trained). This
        # is passed as an explicit parameter below, so it overrides the config
        # file -- keep the default at 0.0 or it silently wins.
        DeclareLaunchArgument("control_rate_hz", default_value="0.0"),
        DeclareLaunchArgument("imu", default_value="true",
                              description="also start the IMU driver"),
        # The manufacturer's node defaults to /dev/imu_usb @ 9600. Surfaced
        # here so a different port does not mean editing a vendored package.
        DeclareLaunchArgument("imu_port", default_value="/dev/imu_usb"),
        DeclareLaunchArgument("imu_baud", default_value="9600"),

        # Nothing below starts unless every joint has a current calibration.
        calibration_gate(strict=True),

        Node(
            package="wit_ros2_imu", executable="wit_ros2_imu",
            name="imuDriverNode", output="log",   # 'log': it clears the screen
            condition=IfCondition(LaunchConfiguration("imu")),
            parameters=[{
                "port": LaunchConfiguration("imu_port"),
                "baudrate": ParameterValue(LaunchConfiguration("imu_baud"),
                                           value_type=int),
            }],
        ),
        Node(
            package="humanoid_deploy", executable="policy_node",
            name="humanoid_policy", output="screen",
            parameters=[config, {"bundle": bundle}],
        ),
        Node(
            package="humanoid_deploy", executable="servo_node",
            name="humanoid_servo", output="screen",
            parameters=[config, {"bundle": bundle, "dry_run": dry_run,
                                 "control_rate_hz": rate}],
        ),
    ])
