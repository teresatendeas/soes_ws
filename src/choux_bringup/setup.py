from setuptools import setup

package_name = 'choux_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/choux_bringup.launch.py']),
        ('share/' + package_name + '/config', ['config/extruder.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kenrich Heavenly Sandria',
    maintainer_email='you@example.com',
    description='Bringup (launch + config) for Choux extruder',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
