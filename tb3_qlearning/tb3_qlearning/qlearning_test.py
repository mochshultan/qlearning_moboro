#!/usr/bin/env python3

import json
import math
import os
import random
import sys
import time

import numpy
import rclpy
from rclpy.node import Node

from turtlebot3_msgs.srv import Dqn


class QLearningTest(Node):

    def __init__(self, stage, load_episode, model_path=None, target_file=None):
        super().__init__('qlearning_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)
        self.custom_model_path = model_path

        self.state_size = 8  # 2 goal + 6 lidar sensors
        self.action_size = 5
        
        # State discretization parameters (sesuai dengan agent)
        self.distance_bins = 3
        self.angle_bins = 10
        self.obstacle_bins = 2

        self.q_table = {}
        self.load_qlearning_model()
        
        # Load target coordinates
        self.target_list = self.load_target_coordinates(target_file)
        self.current_target_index = 0

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.run_test()

    def discretize_state(self, state):
        """Discretize state: sama dengan qlearning_agent.py"""
        state = numpy.asarray(state).flatten()
        
        # Goal distance: 3 bins (>3.0=2, 3.0-1.5=1, 1.5-0=0)
        dist = state[0]
        if dist > 3.0:
            goal_dist = 2
        elif dist > 1.5:
            goal_dist = 1
        else:
            goal_dist = 0
        
        # Goal angle: 10 bins (5 depan + 5 belakang)
        angle_rad = state[1]
        angle_deg = math.degrees(angle_rad)
        
        if -90 <= angle_deg <= 90:  # Area depan: 5 bins
            goal_angle = int((angle_deg + 90) / 36)
            goal_angle = max(0, min(goal_angle, 4))
        else:  # Area belakang: 5 bins
            if angle_deg > 90:
                goal_angle = 5 + int((angle_deg - 90) / 30)
            else:
                goal_angle = 7 + int((angle_deg + 180) / 30)
            goal_angle = max(5, min(goal_angle, 9))
        
        # 6 sensor lidar depan (270°-90°)
        scan_ranges = state[2:]
        obs_bins = tuple(
            1 if scan_ranges[i] > 0.25 else 0
            for i in range(6)
        )
        
        return (goal_dist, goal_angle) + obs_bins

    def get_action(self, state):
        state_key = self.discretize_state(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size

        q_values = self.q_table[state_key]
        max_q = max(q_values)
        best_actions = [a for a in range(self.action_size) if q_values[a] == max_q]

        return random.choice(best_actions)

    def load_target_coordinates(self, target_file=None):
        """Load target coordinates dari file"""
        if target_file is None:
            # Default path
            target_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'model_target.txt'
            )
        
        targets = []
        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        coords = line.split()
                        if len(coords) >= 2:
                            x = float(coords[0])
                            y = float(coords[1])
                            targets.append((x, y))
            self.get_logger().info(f'Loaded {len(targets)} target coordinates from: {target_file}')
        else:
            self.get_logger().warn(f'Target file not found: {target_file}, using default targets')
            # Default targets jika file tidak ada
            targets = [(0.5, 0.0), (1.0, 1.0), (-1.0, 1.0), (1.5, -1.5)]
        
        return targets
    
    def load_qlearning_model(self):
        # Jika custom path diberikan, gunakan itu
        if self.custom_model_path:
            q_table_path = self.custom_model_path
        else:
            # Default path
            model_dir_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                'saved_model'
            )
            q_table_path = os.path.join(
                model_dir_path,
                'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '_qtable.json'
            )

        if os.path.exists(q_table_path):
            with open(q_table_path, 'r') as f:
                q_table_serializable = json.load(f)
                self.q_table = {
                    eval(state_key): q_values
                    for state_key, q_values in q_table_serializable.items()
                }
            self.get_logger().info(f'Q-Learning model loaded from: {q_table_path}')
            self.get_logger().info(f'Q-table size: {len(self.q_table)} states')
        else:
            self.get_logger().error(f'Q-table file not found: {q_table_path}')
            raise FileNotFoundError(f'Model file not found: {q_table_path}')

    def run_test(self):
        success_count = 0
        total_steps = 0
        results = []
        
        self.get_logger().info('='*60)
        self.get_logger().info('Starting Q-Learning Test Mode (Pure Exploitation)')
        self.get_logger().info(f'Loaded Q-table with {len(self.q_table)} states')
        self.get_logger().info(f'Testing with {len(self.target_list)} target coordinates')
        self.get_logger().info('No timeout - Episode ends only on goal reached or collision')
        self.get_logger().info('='*60)
        
        # Test: 1 episode per target, no repeat
        for episode_num, (target_x, target_y) in enumerate(self.target_list, 1):
            self.get_logger().info(f'\n--- Episode {episode_num}/{len(self.target_list)} ---')
            self.get_logger().info(f'Target: ({target_x:.2f}, {target_y:.2f})')
            
            done = False
            score = 0
            local_step = 0
            state = None

            time.sleep(1.0)
            
            # Reset environment dengan target custom
            self.get_logger().info(f'Resetting environment with target ({target_x:.2f}, {target_y:.2f})')
            req_reset = Dqn.Request()
            req_reset.action = 0
            req_reset.init = True
            req_reset.target_x = target_x
            req_reset.target_y = target_y
            
            while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('rl_agent interface service not available, waiting...')
            
            future_reset = self.rl_agent_interface_client.call_async(req_reset)
            rclpy.spin_until_future_complete(self, future_reset)
            
            if future_reset.done() and future_reset.result() is not None:
                state = future_reset.result().state
                self.get_logger().info(f'Environment reset complete, initial distance: {state[0]:.2f}m')
            else:
                self.get_logger().error('Failed to reset environment')
                continue
            
            time.sleep(0.5)

            # Episode loop - no timeout, hanya goal atau collision
            while not done:
                local_step += 1
                
                # Get action from Q-table (pure exploitation, no exploration)
                action = self.get_action(state)

                req = Dqn.Request()
                req.action = action
                req.init = False

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn(
                        'rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    score += reward
                    
                    # Log setiap 100 steps
                    if local_step % 100 == 0:
                        self.get_logger().info(f'  Step: {local_step}, Score: {score:.2f}')
                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')
                    break

                time.sleep(0.05)  # Sesuai dengan agent
            
            # Episode selesai
            total_steps += local_step
            
            # Determine success (goal reached = +200, collision/timeout = -100)
            is_success = score >= 150
            if is_success:
                success_count += 1
            
            result = 'SUCCESS' if is_success else 'FAILED'
            results.append({
                'episode': episode_num,
                'target': (target_x, target_y),
                'success': is_success,
                'steps': local_step,
                'score': score
            })
            
            self.get_logger().info(
                f'Episode {episode_num} {result} | '
                f'Score: {score:.2f} | Steps: {local_step}'
            )
            
            time.sleep(2.0)  # Delay antar episode
        
        # Test selesai - tampilkan summary
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('TEST COMPLETED')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Total Episodes: {len(self.target_list)}')
        self.get_logger().info(f'Success: {success_count}/{len(self.target_list)} ({success_count/len(self.target_list)*100:.1f}%)')
        self.get_logger().info(f'Failed: {len(self.target_list)-success_count}/{len(self.target_list)}')
        self.get_logger().info(f'Total Steps: {total_steps}')
        self.get_logger().info(f'Average Steps: {total_steps/len(self.target_list):.1f}')
        
        # Detail per episode
        self.get_logger().info('\nDetailed Results:')
        for r in results:
            status = 'SUCCESS' if r['success'] else 'FAILED '
            self.get_logger().info(
                f"  Ep {r['episode']}: {status} | "
                f"Target ({r['target'][0]:.2f}, {r['target'][1]:.2f}) | "
                f"Steps: {r['steps']} | Score: {r['score']:.2f}"
            )
        
        self.get_logger().info('='*60)
        self.get_logger().info('Test finished. Press Ctrl+C to exit.')
        
        # Keep node alive untuk melihat hasil
        while rclpy.ok():
            time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args if args else sys.argv)
    
    # Parse arguments
    stage = sys.argv[1] if len(sys.argv) > 1 else '2'
    load_episode = sys.argv[2] if len(sys.argv) > 2 else '1000'
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    target_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("="*60)
    print("Q-Learning Test Mode")
    print("="*60)
    print(f"Stage: {stage}")
    print(f"Episode: {load_episode}")
    if model_path:
        print(f"Custom Model Path: {model_path}")
    else:
        print(f"Default Model Path: saved_model/stage{stage}_episode{load_episode}_qtable.json")
    if target_file:
        print(f"Custom Target File: {target_file}")
    else:
        print(f"Default Target File: model_target.txt")
    print("="*60)
    
    try:
        node = QLearningTest(stage, load_episode, model_path, target_file)
        rclpy.spin(node)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nUsage:")
        print("  ros2 run tb3_qlearning qlearning_test <stage> <episode> [model_path] [target_file]")
        print("\nExamples:")
        print("  ros2 run tb3_qlearning qlearning_test 2 1000")
        print("  ros2 run tb3_qlearning qlearning_test 2 1000 /path/to/custom_qtable.json")
        print("  ros2 run tb3_qlearning qlearning_test 2 1000 None /path/to/custom_targets.txt")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()