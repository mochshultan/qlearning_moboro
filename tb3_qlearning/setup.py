from setuptools import setup
import os
from glob import glob

package_name = 'tb3_qlearning'

setup(
    name=package_name,
    version='0.0.0',
    packages=['tb3_qlearning'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools', 'numpy', 'pyqtgraph', 'PyQt5'],
    zip_safe=True,
    maintainer='mochshultan',
    maintainer_email='shultanalis2004@gmail.com',
    description='Traditional Q-learning agent for TurtleBot3 navigation in Gazebo',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'qlearning_agent = tb3_qlearning.qlearning_agent:main',
            'qlearning_environment = tb3_qlearning.qlearning_environment:main',
            'qlearning_gazebo = tb3_qlearning.qlearning_gazebo:main',
            'qlearning_test = tb3_qlearning.qlearning_test:main',
            'plot_rewards = tb3_qlearning.plot_rewards:main',
        ],
    },
)
