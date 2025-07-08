from setuptools import setup, find_packages


install_requires = [
    'opencv-python',
    ]

setup(
    name="Switch2SMPL",
    version=1.0,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)