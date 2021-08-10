from setuptools import setup

package_name = 'lane_following'

setup(
    name=package_name,
    version='0.0.1',
    packages=[
        'train',
        'polyfuzz'
    ],
    py_modules=[
        'collect',
        'drive',
        'drive_perception',
        'process'
    ],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='David Uhm',
    author_email='david.uhm@lge.com',
    maintainer='David Uhm',
    maintainer_email='david.uhm@lge.com',
    keywords=['ROS',
              'ROS2',
              'deep learning',
              'lane following',
              'end to end',
              'SVL Simulator',
              'Autonomous Driving'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='ROS2-based End-to-End Lane Following model',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collect = collect:main',
            'drive = drive:main',
            'drive_perception = drive_perception:main'
        ],
    },
)
