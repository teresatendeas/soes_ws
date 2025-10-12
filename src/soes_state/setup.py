from setuptools import setup

package_name = 'soes_state'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teresatendeas',
    maintainer_email='teresatendeas@gmail.com',
    description='State machine for SOES system controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_node = soes_state.node:main',
        ],
    },
)
