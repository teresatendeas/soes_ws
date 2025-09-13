from setuptools import setup

package_name = 'choux_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='Kenrich Heavenly Sandria',
    maintainer_email='you@example.com',
    description='Extruder control node for choux paste (stepper backend: sim/serial)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'extruder_node = choux_control.extruder_node:main',
        ],
    },
)
