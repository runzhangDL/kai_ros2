"""The IMU driver, alone, in its own process group.

    ros2 launch humanoid_deploy imu.launch.py

Exists so the IMU can outlive a Ctrl-C aimed at the walking node. Ctrl-C is
delivered to every process in a launch's group, and the walking deployment
deliberately survives it in order to hand the robot back to the standing
policy -- which it can only do while something is still reporting which way is
down. Running the driver from a separate terminal is what makes that true.

Leave this running across as many walking runs as you like; it holds no robot
state.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("imu_port", default_value="/dev/imu_usb"),
        DeclareLaunchArgument("imu_baud", default_value="9600"),
        Node(
            package="wit_ros2_imu", executable="wit_ros2_imu",
            name="imuDriverNode", output="log",
            parameters=[{
                "port": LaunchConfiguration("imu_port"),
                "baudrate": ParameterValue(LaunchConfiguration("imu_baud"),
                                           value_type=int),
            }],
        ),
    ])
