"""Template for any launch file that starts motion nodes.

Copy this pattern into your own bring-up / walking launch files. The single
important line is ``calibration_gate()``: it is evaluated before any process
is spawned, so on an uncalibrated robot the launch aborts with a clear message
and nothing ever starts.

Try it::

    ros2 launch humanoid_calibration guarded_bringup.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node

from humanoid_calibration.launch_guard import calibration_gate


def generate_launch_description():
    return LaunchDescription(
        [
            # ---- the gate --------------------------------------------------
            # Aborts the launch unless every joint in joint_limits.yaml has a
            # current calibration. strict=True ignores the
            # HUMANOID_ALLOW_UNCALIBRATED escape hatch, which is what you want
            # for anything that moves the legs.
            calibration_gate(strict=True),
            # ---- observability ---------------------------------------------
            Node(
                package="humanoid_calibration",
                executable="calibration_status",
                name="calibration_status",
                output="screen",
            ),
            # ---- your motion nodes go below --------------------------------
            # Node(package='humanoid_walk', executable='walking',
            #      name='walking', output='screen'),
            # Node(package='humanoid_hw', executable='servo_bus',
            #      name='servo_bus', output='screen'),
        ]
    )
