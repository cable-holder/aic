from glob import glob

from setuptools import find_packages, setup

package_name = "ch_inference"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yl",
    maintainer_email="yl@example.com",
    description="LeRobot VLA inference policy for AIC cable insertion.",
    license="Apache-2.0",
)
