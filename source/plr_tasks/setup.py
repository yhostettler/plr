from setuptools import find_packages, setup

setup(
    name="plr_tasks",
    version="0.1.0",
    description="PLR custom Isaac Lab tasks",
    author="PLR team",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
