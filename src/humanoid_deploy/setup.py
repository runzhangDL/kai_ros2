from glob import glob
import os

from setuptools import find_packages, setup

package_name = "humanoid_deploy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tools"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # The exported bundle is what the robot actually loads; the .pkl is kept
        # alongside it so the export can always be reproduced.
        (os.path.join("share", package_name, "models"),
         glob("models/*.npz") + glob("models/*.pkl")),
    ],
    install_requires=["setuptools", "numpy", "pyserial"],
    zip_safe=True,
    maintainer="run",
    maintainer_email="runzhang715@gmail.com",
    description=(
        "Deployment of the trained standing policy: numpy inference, IMU "
        "conditioning, calibrated joint mapping and the safety supervisor."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = humanoid_deploy.policy_node:main",
            "servo_node = humanoid_deploy.servo_node:main",
            # Walking runs the executor and both policies in ONE process, so
            # that a Ctrl-C can hand the robot back to the standing policy
            # instead of killing whatever is holding it up.
            "walk_node = humanoid_deploy.walk_node:main",
            "bus_benchmark = humanoid_deploy.bus_benchmark:main",
        ],
    },
)
