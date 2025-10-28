from setuptools import setup
from glob import glob
import os

package_name = 'soes_bringup'

setup(
    entry_points={'console_scripts': [
            'soes_comm_i2c = soes_state.soes_comm_i2c:main',
            'one_motor_cmd = soes_state.one_motor_cmd:main',]},
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teresatendeas',
    maintainer_email='teresatendeas@gmail.com',
    description='SOES bringup with centralized configs',
    license='MIT',
)
