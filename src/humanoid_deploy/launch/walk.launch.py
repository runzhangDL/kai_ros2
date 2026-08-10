"""Bring up the stand -> walk -> stand deployment.

    # terminal 1: the IMU, on its own so Ctrl-C below cannot kill it
    ros2 launch humanoid_deploy imu.launch.py

    # terminal 2: dry run -- nothing is written to a servo
    ros2 launch humanoid_deploy walk.launch.py \\
        bundle:=/path/policy_bundle.npz walk_bundle:=/path/walk_bundle.npz

    # once the dry run looks right
    ros2 launch humanoid_deploy walk.launch.py bundle:=... walk_bundle:=... \\
        dry_run:=false

Then, holding the robot upright:

    ros2 service call /humanoid_servo/arm  std_srvs/srv/Trigger   # it stands
    ros2 service call /humanoid_policy/walk std_srvs/srv/Trigger  # it walks

It crouches, walks for ``walk_duration_s``, and returns to standing by itself.
Ctrl-C in terminal 2 during a walk does the same thing early: the robot goes
back to the standing policy and stays up. A second Ctrl-C exits and drops
torque, so hold the robot before pressing it.

Why the IMU is not started here by default
------------------------------------------
Ctrl-C goes to every process in the launch's group. The walking node survives
it on purpose -- that is the whole point -- but the IMU driver does not, and a
handback flying on a frozen gravity vector is not a handback. Starting the IMU
from its own terminal keeps it alive across the interrupt. ``imu:=true`` is
available for convenience on runs where that does not matter (a dry run, or
one that will only ever end by the duration timer).
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
    deploy_config = os.path.join(share, "config", "deploy.yaml")
    walk_config = os.path.join(share, "config", "walk.yaml")

    bundle = LaunchConfiguration("bundle")
    walk_bundle = LaunchConfiguration("walk_bundle")

    return LaunchDescription([
        DeclareLaunchArgument("bundle", description="standing policy_bundle.npz"),
        DeclareLaunchArgument("walk_bundle", description="walking walk_bundle.npz"),
        # Both configs, in order: walk.yaml carries only the differences and
        # must win, so it is listed second.
        DeclareLaunchArgument("config", default_value=deploy_config),
        DeclareLaunchArgument("walk_config", default_value=walk_config),
        DeclareLaunchArgument("dry_run", default_value="true"),
        # 0.0 = the rate recorded in the bundle (25 Hz, as trained). Passed
        # explicitly below, so it overrides the config file -- keep it at 0.0.
        DeclareLaunchArgument("control_rate_hz", default_value="0.0"),
        DeclareLaunchArgument("walk_duration_s", default_value="5.0"),
        # See the module docstring: leave this false and run imu.launch.py in
        # its own terminal for any run whose handback has to work.
        DeclareLaunchArgument("imu", default_value="false"),
        DeclareLaunchArgument("imu_port", default_value="/dev/imu_usb"),
        DeclareLaunchArgument("imu_baud", default_value="9600"),

        calibration_gate(strict=True),

        Node(
            package="wit_ros2_imu", executable="wit_ros2_imu",
            name="imuDriverNode", output="log",
            condition=IfCondition(LaunchConfiguration("imu")),
            parameters=[{
                "port": LaunchConfiguration("imu_port"),
                "baudrate": ParameterValue(LaunchConfiguration("imu_baud"),
                                           value_type=int),
            }],
        ),
        # ONE process: the executor and both policies. See walk_node.main --
        # splitting them would mean a Ctrl-C that kills the only thing holding
        # the robot up.
        Node(
            package="humanoid_deploy", executable="walk_node",
            # No `name=`: this process hosts two nodes, humanoid_servo and
            # humanoid_policy, and a launch-level name would rename both --
            # taking /humanoid_servo/arm and /humanoid_policy/walk with it, and
            # detaching them from their sections in the config files.
            # `exec_name` labels the process without touching node names.
            exec_name="humanoid_walk", output="screen",
            parameters=[
                LaunchConfiguration("config"),
                LaunchConfiguration("walk_config"),
                {"bundle": bundle,
                 "walk_bundle": walk_bundle,
                 "dry_run": LaunchConfiguration("dry_run"),
                 "control_rate_hz": LaunchConfiguration("control_rate_hz"),
                 "walk_duration_s": ParameterValue(
                     LaunchConfiguration("walk_duration_s"), value_type=float)},
            ],
        ),
    ])
