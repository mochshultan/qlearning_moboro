#!/usr/bin/env python3

import collections
import datetime
import json
import math
import os
import random
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn  # Keep for ROS interface compatibility


LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')


class QLearningAgent(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('qlearning_agent')

        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 8  # 2 goal + 6 lidar sensors
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)

        # Q-Learning hyperparameters (optimized for faster convergence)
        self.learning_rate = 0.2      # α: higher for faster learning
        self.discount_factor = 0.99   # γ: lower for more immediate rewards
        self.epsilon = 1.0            # ε: exploration rate
        self.epsilon_decay = 0.999    # ε: faster decay for exploitation
        self.epsilon_min = 0.05       # minimum ε: lower for more exploitation

        # Q-table: discretized state space to Q-values
        self.q_table = {}
        self.distance_bins = 3    # goal distance bins: >3.0, 3.-1.5, 1.5-0
        self.angle_bins = 10      # goal angle bins: 5 depan + 5 belakang
        self.obstacle_bins = 2    # obstacle distance bins (close/far, threshold 0.25) for 6 sensors
        # State space: 3 × 10 × 2^6 = 1920 states

        self.load_model = False
        self.load_episode = 0
        self.load_from_stage = 2  # Load Q-table dari stage ini

        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        self.model_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.load_from_stage) + '_episode' + str(self.load_episode) + '.json'
        )

        # Create saved_model directory if it doesn't exist
        os.makedirs(self.model_dir_path, exist_ok=True)

        if self.load_model:
            self.load_qlearning_model()

        if LOGGING:
            # Simple logging for traditional Q-Learning
            log_file_name = f'qlearning_stage{self.stage}_rewards.log'
            home_dir = os.path.expanduser('~')
            log_dir = os.path.join(home_dir, 'qlearning_logs')
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, log_file_name)
            self.log_file = open(self.log_file_path, 'a')

        # ROS2 service clients (keep for environment interface)
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0

            time.sleep(0.1)  # Reduced delay for faster movement

            while True:
                local_step += 1

                action = int(self.get_action(state))
                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode and next_state is not None:
                    self.train_qlearning(state, action, reward, next_state, done)

                state = next_state

                if done or next_state is None:
                    msg = Float32MultiArray()
                    msg.data = [float(score), 0.0]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        state_space = self.distance_bins * self.angle_bins * (self.obstacle_bins ** 6)
                        self.log_file.write(
                            f'Episode: {episode_num}, Score: {score}, '
                            f'Q-table size: {len(self.q_table)}, '
                            f'Epsilon: {self.epsilon:.4f}, '
                            f'State space: {state_space}\n'
                        )
                        self.log_file.flush()

                    state_space = self.distance_bins * self.angle_bins * (self.obstacle_bins ** 6)
                    print(
                        'Episode:', episode,
                        'score:', score,
                        'q_table size:', len(self.q_table),
                        'epsilon:', f'{self.epsilon:.4f}',
                        'state_space:', f'{state_space}',
                        'coverage:', f'{len(self.q_table)}/{state_space} ({100*len(self.q_table)/state_space:.1f}%)')

                    param_keys = ['epsilon', 'episode']
                    param_values = [self.epsilon, episode]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                time.sleep(0.05)  # 50ms delay between steps, nyesuaikan yg jundi

            if self.train_mode:
                # Decay epsilon after each episode
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # Save more frequently due to larger state space
                if episode % 50 == 0:
                    self.save_qlearning_model(episode, param_dictionary)
                    


        if LOGGING:
            self.log_file.close()

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )

        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )

        future = self.reset_environment_client.call_async(Dqn.Request())
        state = None

        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            state = future.result().state
            # Traditional Q-Learning: no batch dimension needed
            state = numpy.asarray(state).flatten()
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state

    def discretize_state(self, state):
        """Discretize state: 6 front lidar sensors (270°-90°), 10 angle bins (5 depan + 5 belakang), 3 distance bins"""
        # Goal distance: 3 bins (>3.0=2, 3.0-1.5=1, 1.5-0=0)
        dist = state[0]
        if dist > 3.0:
            goal_dist = 2
        elif dist > 1.5:
            goal_dist = 1
        else:
            goal_dist = 0
        
        # Goal angle: 10 bins (5 depan: -90° hingga +90°, 5 belakang: 90°-180° dan -180° hingga -90°)
        angle_rad = state[1]  # [-π, π]
        angle_deg = math.degrees(angle_rad)  # [-180, 180]
        
        if -90 <= angle_deg <= 90:  # Area depan: 5 bins
            # Bin 0-4: -90° hingga +90°, setiap bin = 36°
            goal_angle = int((angle_deg + 90) / 36)
            goal_angle = max(0, min(goal_angle, 4))
        else:  # Area belakang: 5 bins
            # Bin 5-9: 90°-180° dan -180° hingga -90°
            if angle_deg > 90:
                # 90°-180° → bin 5-7
                goal_angle = 5 + int((angle_deg - 90) / 30)
            else:
                # -180° hingga -90° → bin 7-9
                goal_angle = 7 + int((angle_deg + 180) / 30)
            goal_angle = max(5, min(goal_angle, 9))
        
        # 6 sensor lidar depan (270°-90°), sudah difilter di environment
        scan_ranges = state[2:]  # 6 sensor readings
        obs_bins = tuple(
            1 if scan_ranges[i] > 0.25 else 0  # 1=jauh, 0=dekat (threshold 0.25)
            for i in range(6)
        )
        
        return (goal_dist, goal_angle) + obs_bins

    def get_action(self, state):
        """ε-greedy action selection"""
        if self.train_mode and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: best action
            return self.get_best_action(state)

    def get_best_action(self, state):
        state_key = self.discretize_state(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size

        q_values = self.q_table[state_key]
        max_q = max(q_values)
        best_actions = [a for a in range(self.action_size) if q_values[a] == max_q]

        return random.choice(best_actions)

    def step(self, action):
        req = Dqn.Request()
        req.action = action
        req.init = False
        req.target_x = 0.0
        req.target_y = 0.0

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)
        next_state = None
        reward = 0.0
        done = True

        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            next_state = future.result().state
            # Traditional Q-Learning: no batch dimension needed
            next_state = numpy.asarray(next_state).flatten()
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return next_state, reward, done

    def train_qlearning(self, state, action, reward, next_state, done):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.action_size

        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key]) if not done else 0.0
        
        # Q-Learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        target = reward + self.discount_factor * max_next_q
        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)

    def save_qlearning_model(self, episode, params):
        q_table_serializable = {}
        for state_key, q_values in self.q_table.items():
            q_table_serializable[str(state_key)] = q_values

        q_table_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(episode) + '_qtable.json'
        )
        with open(q_table_path, 'w') as f:
            json.dump(q_table_serializable, f)

        params_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(episode) + '_params.json'
        )
        with open(params_path, 'w') as f:
            json.dump(params, f)

        print(f'Q-table saved at episode {episode}')

    def load_qlearning_model(self):
        q_table_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '_qtable.json'
        )
        params_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '_params.json'
        )

        if os.path.exists(q_table_path):
            with open(q_table_path, 'r') as f:
                q_table_serializable = json.load(f)
                self.q_table = {
                    eval(state_key): q_values
                    for state_key, q_values in q_table_serializable.items()
                }

        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                self.epsilon = params.get('epsilon', 1.0)

        print(f'Q-table loaded from episode {self.load_episode}')
        print(f'Q-parameters loaded from episode {self.load_episode}')


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '2'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    rclpy.init(args=args)

    qlearning_agent = QLearningAgent(stage_num, max_training_episodes)
    qlearning_agent.process()

    qlearning_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()