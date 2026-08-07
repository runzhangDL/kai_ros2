from glob import glob
import os

from setuptools import find_packages, setup

package_name = "humanoid_calibration"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "pyyaml", "pyserial"],
    zip_safe=True,
    maintainer="run",
    maintainer_email="runzhang715@gmail.com",
    description=(
        "One-time read-only zero and travel-limit calibration for the humanoid's "
        "Feetech STS3215 servos, plus the uncalibrated-robot guard."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "calibrate = humanoid_calibration.calibrate_cli:main",
            "verify_calibration = humanoid_calibration.verify_cli:main",
            "check_calibration = humanoid_calibration.guard:main",
            "calibration_status = humanoid_calibration.calibration_status_node:main",
        ],
    },
)
