#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import math
import os

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal


ROS_DISTRO = os.environ.get('ROS_DISTRO', 'humble')  # Default to humble if not set


class RLEnvironment(Node):

    def __init__(self):
        super().__init__('rl_environment')
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        self.max_step = 2000

        self.done = False
        self.fail = False
        self.succeed = False

        self.thres_goal = 0.25  # Threshold to consider goal reached
        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.scan_ranges = []
        self.front_ranges = []
        self.min_obstacle_distance = 10.0
        self.is_front_min_actual_front = False

        self.local_step = 0
        self.stop_cmd_vel_timer = None
        self.prev_goal_distance = 0.0  # Initialize to prevent AttributeError
        # Higher velocity values for faster movement
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]  # Increased angular velocities
        self.linear_vel = 0.2                            # Linear velocities for each action
        
        # Logging per step
        self.episode_num = 0
        home_dir = os.path.expanduser('~')
        log_dir = os.path.join(home_dir, 'qlearning_logs')
        os.makedirs(log_dir, exist_ok=True)
        self.step_log_path = os.path.join(log_dir, 'step_details.log')
        self.step_log_file = open(self.step_log_path, 'a')
        print(f'Step log file created at: {self.step_log_path}')

        qos = QoSProfile(depth=10)

        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_sub_callback,
            qos_profile_sensor_data
        )

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            'task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            'task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            'initialize_env',
            callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Dqn,
            'rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            'make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn,
            'reset_environment',
            self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        self.get_logger().info('Make environment called')
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'service for initialize the environment is not available, waiting ...'
            )
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal.success:
            self.get_logger().error('initialize environment request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(
                'goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y)
            )

        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        self.episode_num += 1
        self.local_step = 0
        response.state = state

        return response

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task succeed finished')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task failed finished')
        else:
            self.get_logger().error('task failed service call failed')

    def scan_sub_callback(self, scan):
        # 22 samples total, ambil 11 sensor depan (270°-90°)
        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        all_ranges = []
        self.front_ranges = []

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = scan.ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            all_ranges.append(distance)

            # Filter 11 sensor depan: 270°-90° (atau -90° hingga +90° dalam radian)
            angle_deg = math.degrees(angle) % 360
            if angle_deg >= 270 or angle_deg <= 90:
                self.front_ranges.append(distance)

        self.scan_ranges = self.front_ranges  # Hanya 11 sensor depan
        self.min_obstacle_distance = min(all_ranges) if all_ranges else 10.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        state = [float(self.goal_distance), float(self.goal_angle)]
        # Tambahkan 11 sensor lidar depan (270°-90°)
        for var in self.scan_ranges:
            state.append(float(var))
        self.local_step += 1
        return state
    
    def check_episode_end(self):
        """Check if episode should end and handle service calls"""
        if self.goal_distance < self.thres_goal:  # Larger goal radius to make it easier
            self.get_logger().info('Goal Reached')
            self.succeed = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_succeed()

        elif self.min_obstacle_distance < 0.15:
            self.get_logger().info('Collision happened')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        elif self.local_step == self.max_step:
            self.get_logger().info('Time out!')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

    def calculate_reward(self):
        if self.succeed:
            return 200.0, 0.0, 0.0  # reward_total, Rd, Rtheta
        elif self.fail:
            return -100.0, 0.0, 0.0
        
        # --- 1. Reward Jarak (Differential) ---
        d_old = self.prev_goal_distance
        d_new = self.goal_distance
        distance_rate = d_old - d_new
        Rd = 100.0 * distance_rate 
        
        # --- 2. Reward Sudut (Heading) ---
        e_theta = self.goal_angle 
        e_theta_normalized = math.atan2(math.sin(e_theta), math.cos(e_theta))
        Rtheta = 0.5 * (1.0 - 2.0 * abs(e_theta_normalized) / math.pi)
        
        # --- 3. Total ---
        reward = Rd + Rtheta
        
        # Update
        self.prev_goal_distance = self.goal_distance
        
        return reward, Rd, Rtheta
    
    def rl_agent_interface_callback(self, request, response):
        # Handle reset dengan target custom
        if request.init:
            self.get_logger().info(f'Resetting with custom target: ({request.target_x:.2f}, {request.target_y:.2f})')
            self.goal_pose_x = request.target_x
            self.goal_pose_y = request.target_y
            self.local_step = 0
            self.done = False
            self.succeed = False
            self.fail = False
            # Stop robot
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            # Return initial state
            response.state = self.calculate_state()
            self.init_goal_distance = response.state[0]
            self.prev_goal_distance = self.init_goal_distance
            response.reward = 0.0
            response.done = False
            return response
        
        action = request.action
        if ROS_DISTRO == 'humble':
            msg = Twist()
            # Action-specific linear velocity mapping
            if action == 0 or action == 4:  # Pure rotation actions
                msg.linear.x = 0.0  # No forward motion
            else:
                msg.linear.x = self.linear_vel  # forward and moderate forward with turning
            msg.angular.z = self.angular_vel[action]
        else:
            msg = TwistStamped()
            # Action-specific linear velocity mapping
            if action == 0 or action == 4:  # Pure rotation actions
                msg.twist.linear.x = 0.0  # No forward motion
            else:
                msg.twist.linear.x = self.linear_vel  # forward and moderate forward with turning
            msg.twist.angular.z = self.angular_vel[action]

        self.cmd_vel_pub.publish(msg)
        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)

        response.state = self.calculate_state()
        self.check_episode_end()  # Check episode end conditions once
        reward_total, Rd, Rtheta = self.calculate_reward()
        response.reward = reward_total
        response.done = self.done
        
        # Log per step: step, Rd, Rtheta, reward_step, min_obstacle
        log_line = (
            f'Step: {self.local_step}, '
            f'Rd: {Rd:.2f}, Rtheta: {Rtheta:.2f}, '
            f'R_step: {reward_total:.2f}, MinObs: {self.min_obstacle_distance:.3f}\n'
        )
        self.step_log_file.write(log_line)
        self.step_log_file.flush()
        # Print setiap 10 step untuk monitoring
        if self.local_step % 10 == 0:
            print(f'[LOG] {log_line.strip()}')

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        self.get_logger().info('Stop called')
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())
        self.destroy_timer(self.stop_cmd_vel_timer)
        self.stop_cmd_vel_timer = None  # Reset to None after destroying

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    rl_environment = RLEnvironment()
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()