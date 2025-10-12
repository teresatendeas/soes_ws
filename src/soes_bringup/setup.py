from setuptools import setup

package_name = 'soes_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bringup.launch.py']),
        ('share/' + package_name + '/config', [
            'config/state.yaml',
            'config/robothand.yaml',
            'config/vision.yaml'
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teresatendeas',
    maintainer_email='teresatendeas@gmail.com',
    description='SOES bringup and launch configuration package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
