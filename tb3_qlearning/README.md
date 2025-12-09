# Q-Learning for TurtleBot3 Navigation

Implementasi algoritma **Q-Learning** untuk navigasi otonom TurtleBot3 di lingkungan simulasi Gazebo dengan obstacle avoidance.

---

## 📋 Daftar Isi
- [Teori Q-Learning](#teori-q-learning)
- [Arsitektur Sistem](#arsitektur-sistem)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Hyperparameters](#hyperparameters)
- [Parameters](#parameters)
- [Q-Table Structure](#q-table-structure)
- [Policy](#policy)
- [Training Process](#training-process)
- [Logging](#logging)

---

## 🎓 Teori Q-Learning

### Bellman Equation
Q-Learning menggunakan **Bellman Optimality Equation** untuk update Q-values:

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                      a'
```

**Komponen:**
- `Q(s,a)`: Q-value untuk state `s` dan action `a`
- `α` (alpha): Learning rate (0.2)
- `r`: Reward yang diterima
- `γ` (gamma): Discount factor (0.99)
- `s'`: Next state
- `max Q(s',a')`: Maximum Q-value untuk next state

### Karakteristik
- **Model-free**: Tidak memerlukan model environment
- **Off-policy**: Belajar dari optimal policy meskipun menggunakan exploratory policy
- **Value-based**: Mempelajari value function (Q-table)
- **Tabular**: Menggunakan lookup table untuk menyimpan Q-values

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────┐
│  qlearning_agent.py │
│  - Q-table          │
│  - ε-greedy policy  │
│  - State discretize │
│  - Q-learning update│
└──────────┬──────────┘
           │ ROS2 Service
           │ (Dqn.srv)
           ▼
┌─────────────────────────┐
│ qlearning_environment.py│
│  - State calculation    │
│  - Reward calculation   │
│  - Action execution     │
│  - Episode management   │
└──────────┬──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Gazebo + TB3 │
    └──────────────┘
```

---

## 📊 State Space

### Raw State (Continuous)
State mentah dari environment:
```python
state = [goal_distance, goal_angle, lidar_0, lidar_1, ..., lidar_5]
# Total: 8 dimensi (2 goal + 6 lidar)
```

### Discretized State
State di-diskritisasi untuk Q-table:

#### 1. **Goal Distance** (3 bins)
| Bin | Range | Deskripsi |
|-----|-------|-----------|
| 0 | 0 - 1.5m | Dekat |
| 1 | 1.5 - 3.0m | Sedang |
| 2 | > 3.0m | Jauh |

#### 2. **Goal Angle** (10 bins)
**Area Depan (5 bins):** -90° hingga +90°
| Bin | Range | Deskripsi |
|-----|-------|-----------|
| 0 | -90° to -54° | Kiri depan |
| 1 | -54° to -18° | Kiri |
| 2 | -18° to +18° | Lurus |
| 3 | +18° to +54° | Kanan |
| 4 | +54° to +90° | Kanan depan |

**Area Belakang (5 bins):** 90°-180° dan -180° hingga -90°
| Bin | Range | Deskripsi |
|-----|-------|-----------|
| 5 | +90° to +120° | Kanan belakang |
| 6 | +120° to +150° | Belakang kanan |
| 7 | +150° to -150° | Belakang |
| 8 | -150° to -120° | Belakang kiri |
| 9 | -120° to -90° | Kiri belakang |

#### 3. **LiDAR Obstacles** (6 sensors × 2 bins)
- **6 sensor depan**: 270° - 90° (filtered dari 12 samples total)
- **2 bins per sensor**: 
  - `0`: Dekat (≤ 0.25m)
  - `1`: Jauh (> 0.25m)

### Total State Space
```
Total States = 3 × 10 × 2^6 = 1,920 states
```

**State Key Format:**
```python
state_key = (goal_dist, goal_angle, obs_0, obs_1, obs_2, obs_3, obs_4, obs_5)
# Contoh: (1, 2, 1, 1, 0, 1, 1, 1)
```

---

## 🎮 Action Space

5 discrete actions untuk kontrol robot:

| Action | Linear Vel (m/s) | Angular Vel (rad/s) | Deskripsi |
|--------|------------------|---------------------|-----------|
| 0 | 0.0 | +1.5 | Rotate left (pure rotation) |
| 1 | 0.2 | +0.75 | Forward + turn left |
| 2 | 0.2 | 0.0 | Forward straight |
| 3 | 0.2 | -0.75 | Forward + turn right |
| 4 | 0.0 | -1.5 | Rotate right (pure rotation) |

**Action Duration:** 0.8 detik per action (timer-based)

---

## 🎁 Reward Function

### Terminal Rewards
| Kondisi | Reward | Trigger |
|---------|--------|---------|
| **Goal Reached** | +200.0 | `goal_distance < 0.25m` |
| **Collision** | -100.0 | `min_obstacle < 0.15m` |
| **Timeout** | -100.0 | `step > 2000` |

### Step Rewards (Non-terminal)
Reward per step terdiri dari 2 komponen:

#### 1. **Distance Reward (Rd)**
```python
Rd = 100.0 × (d_old - d_new)
```
- **Range:** -∞ hingga +∞ (praktis: -10 hingga +10)
- **Positif:** Robot mendekat ke target
- **Negatif:** Robot menjauh dari target
- **Zero:** Robot tidak bergerak (stuck)

#### 2. **Angle Reward (Rθ)**
```python
e_theta = goal_angle  # [-π, π]
e_theta_normalized = atan2(sin(e_theta), cos(e_theta))
Rθ = 0.5 × (1.0 - 2.0 × |e_theta_normalized| / π)
```
- **Range:** -0.5 hingga +0.5
- **+0.5:** Target tepat di depan (0°)
- **0.0:** Target di samping (±90°)
- **-0.5:** Target di belakang (±180°)

#### Total Step Reward
```python
R_total = Rd + Rθ
```
**Range:** Sekitar -10.5 hingga +10.5 per step

---

## ⚙️ Hyperparameters

| Parameter | Symbol | Value | Deskripsi |
|-----------|--------|-------|-----------|
| **Learning Rate** | α | 0.2 | Seberapa cepat Q-values diupdate |
| **Discount Factor** | γ | 0.99 | Pentingnya future rewards |
| **Initial Epsilon** | ε₀ | 1.0 | Exploration rate awal (100%) |
| **Epsilon Decay** | - | 0.999 | Decay rate per episode |
| **Min Epsilon** | ε_min | 0.05 | Minimum exploration (5%) |

### Epsilon Decay Schedule
```python
ε(t) = max(ε_min, ε₀ × decay^t)
```

**Contoh:**
- Episode 1: ε = 1.0 (100% exploration)
- Episode 100: ε ≈ 0.90
- Episode 500: ε ≈ 0.61
- Episode 1000: ε ≈ 0.37
- Episode 2000: ε ≈ 0.13
- Episode 3000+: ε = 0.05 (minimum)

---

## 📐 Parameters

### Environment Parameters
| Parameter | Value | Deskripsi |
|-----------|-------|-----------|
| **Max Steps** | 2000 | Maximum steps per episode |
| **Goal Threshold** | 0.25m | Jarak untuk goal reached |
| **Collision Threshold** | 0.15m | Jarak untuk collision |
| **Linear Velocity** | 0.2 m/s | Kecepatan maju |
| **Angular Velocities** | [1.5, 0.75, 0.0, -0.75, -1.5] rad/s | Kecepatan rotasi |
| **Action Duration** | 0.8s | Durasi eksekusi action |
| **Step Delay** | 0.05s | Delay antar step di agent |

### LiDAR Configuration
| Parameter | Value | Deskripsi |
|-----------|-------|-----------|
| **Total Samples** | 12 | Total LiDAR rays |
| **Used Sensors** | 6 | Sensor depan (270°-90°) |
| **Max Range** | 3.5m | Maximum detection range |
| **Obstacle Threshold** | 0.25m | Threshold dekat/jauh |

### Training Parameters
| Parameter | Value | Deskripsi |
|-----------|-------|-----------|
| **Save Frequency** | 50 episodes | Q-table save interval |
| **Log Frequency** | Every episode | Episode log |
| **Step Log Frequency** | Every 10 steps | Step detail log |

---

## 🗂️ Q-Table Structure

### Format
```python
Q-table = {
    state_key: [Q(s,a₀), Q(s,a₁), Q(s,a₂), Q(s,a₃), Q(s,a₄)]
}
```

### Contoh Entry
```python
state_key = (1, 2, 1, 1, 0, 1, 1, 1)
# goal_dist=1 (sedang), goal_angle=2 (lurus), obstacles=[1,1,0,1,1,1]

Q-values = [2.5, 5.3, 8.7, 4.2, 1.8]
#          [a₀,  a₁,  a₂,  a₃,  a₄]
# Best action: a₂ (forward straight) dengan Q=8.7
```

### Storage
- **Format:** JSON
- **Path:** `saved_model/stage{N}_episode{M}_qtable.json`
- **Size:** Grows dynamically (max 1,920 entries)
- **Initialization:** Lazy (entries created on-demand)

### Q-Table Coverage
```python
Coverage = (Visited States / Total States) × 100%
         = (len(Q-table) / 1920) × 100%
```

**Typical Coverage:**
- Episode 100: ~5-10%
- Episode 500: ~20-30%
- Episode 1000: ~40-50%
- Episode 2000+: ~60-70%

---

## 🎯 Policy

### ε-Greedy Policy
Kombinasi exploration dan exploitation:

```python
if random() < ε:
    action = random_action()  # Exploration
else:
    action = argmax Q(s,a)    # Exploitation
            a
```

### Action Selection Process
1. **Discretize state** → state_key
2. **Check Q-table:**
   - Jika state_key belum ada → initialize dengan [0.0, 0.0, 0.0, 0.0, 0.0]
3. **ε-greedy selection:**
   - Dengan probabilitas ε: pilih random action
   - Dengan probabilitas (1-ε): pilih action dengan Q-value tertinggi
4. **Tie-breaking:** Jika multiple actions punya Q-value sama → pilih random

### Policy Evolution
- **Early training (ε ≈ 1.0):** Mostly exploration, random actions
- **Mid training (ε ≈ 0.5):** Balanced exploration-exploitation
- **Late training (ε ≈ 0.05):** Mostly exploitation, greedy actions

---

## 🔄 Training Process

### Episode Loop
```
FOR each episode:
    1. Reset environment → get initial state
    2. Initialize episode_score = 0
    
    WHILE not done:
        3. Select action using ε-greedy policy
        4. Execute action in environment
        5. Observe next_state, reward, done
        6. Update Q-table using Bellman equation
        7. episode_score += reward
        8. state ← next_state
    
    9. Decay epsilon: ε ← max(ε_min, ε × decay)
    10. Log episode results
    11. Save Q-table (every 50 episodes)
```

### Q-Learning Update
```python
# 1. Discretize states
s_key = discretize(state)
s'_key = discretize(next_state)

# 2. Get current Q-value
Q_current = Q_table[s_key][action]

# 3. Get max Q-value for next state
if done:
    max_next_q = 0.0
else:
    max_next_q = max(Q_table[s'_key])

# 4. Calculate target
target = reward + γ × max_next_q

# 5. Update Q-value
Q_table[s_key][action] = Q_current + α × (target - Q_current)
```

---

## 📊 Logging

### Episode Logs
**Path**: `~/qlearning_logs/qlearning_stage{N}_rewards.log`

**Format**:
```
Episode: 1, Score: 45.23, Q-table size: 12, Epsilon: 0.9990, State space: 1920
Episode: 2, Score: -78.45, Q-table size: 25, Epsilon: 0.9980, State space: 1920
```

### Step Details
**Path**: `~/qlearning_logs/step_details.log`

**Format** (every 10 steps):
```
Step: 10, Rd: 2.34, Rtheta: 0.45, R_step: 2.79, MinObs: 1.234
Step: 20, Rd: -1.23, Rtheta: 0.12, R_step: -1.11, MinObs: 0.876
```

### Saved Models
**Path**: `saved_model/stage{N}_episode{M}_qtable.json`

---

## 🚀 Usage

### Training
```bash
# Terminal 1: Launch Gazebo
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_stage2.launch.py

# Terminal 2: Start training
ros2 run tb3_qlearning qlearning_agent 2 1000

# Terminal 3: (Optional) Live plot
ros2 run tb3_qlearning plot_rewards
```

### Testing
```bash
# Run test with default settings
ros2 run tb3_qlearning qlearning_test 2 1000

# With custom model
ros2 run tb3_qlearning qlearning_test 2 1000 /path/to/qtable.json

# With custom targets
ros2 run tb3_qlearning qlearning_test 2 1000 None /path/to/targets.txt
```

---

## 📚 References

1. **Q-Learning**: Watkins, C.J.C.H. (1989). "Learning from Delayed Rewards"
2. **Bellman Equation**: Bellman, R. (1957). "Dynamic Programming"
3. **ε-greedy**: Sutton & Barto (2018). "Reinforcement Learning: An Introduction"
4. **TurtleBot3**: ROBOTIS e-Manual - https://emanual.robotis.com/
    Q_next_max = 0.0
else:
    Q_next_max = max(Q_table[s'_key])

# 4. Calculate target
target = reward + γ × Q_next_max

# 5. Update Q-value
Q_table[s_key][action] = Q_current + α × (target - Q_current)
```

### Convergence Criteria
Training dianggap converge ketika:
- Q-values stabil (perubahan < threshold)
- Episode rewards konsisten tinggi
- Success rate > 80%
- Epsilon mencapai minimum (0.05)

---

## 📝 Logging

### Episode Log
**File:** `~/qlearning_logs/qlearning_stage{N}_rewards.log`

**Format:**
```
Episode: 1, Score: -45.23, Q-table size: 45, Epsilon: 0.9990, State space: 1920
Episode: 2, Score: 12.67, Q-table size: 78, Epsilon: 0.9980, State space: 1920
...
```

**Fields:**
- `Episode`: Episode number
- `Score`: Total reward untuk episode
- `Q-table size`: Jumlah states yang sudah dikunjungi
- `Epsilon`: Current exploration rate
- `State space`: Total possible states (1920)

### Step Log
**File:** `~/qlearning_logs/step_details.log`

**Format:**
```
Step: 1, Rd: 1.50, Rtheta: 0.35, R_step: 1.85, MinObs: 1.250
Step: 2, Rd: 0.80, Rtheta: 0.42, R_step: 1.22, MinObs: 1.180
...
```

**Fields:**
- `Step`: Step number dalam episode
- `Rd`: Distance reward component
- `Rtheta`: Angle reward component
- `R_step`: Total step reward (Rd + Rtheta)
- `MinObs`: Minimum obstacle distance (m)

### Console Output
```
Episode: 100 score: 156.34 q_table size: 234 epsilon: 0.9048 
state_space: 1920 coverage: 234/1920 (12.2%)
```

---

## 🚀 Usage

### Training
```bash
# Terminal 1: Launch Gazebo + Environment
ros2 launch tb3_qlearning qlearning_stage2.launch.py

# Terminal 2: Run Agent
ros2 run tb3_qlearning qlearning_agent 2 5000

# Terminal 3: (Optional) Live Graph
ros2 run tb3_qlearning plot_rewards
```

### Load Pretrained Model
```python
self.load_model = True
self.load_episode = 1000
self.load_from_stage = 2
```

### Evaluation Mode
```python
self.train_mode = False  # Disable training
self.epsilon = 0.0       # Pure exploitation
```

---

## 📈 Performance Metrics

### Success Metrics
- **Success Rate:** % episodes yang mencapai goal
- **Average Steps:** Rata-rata steps untuk mencapai goal
- **Average Reward:** Rata-rata total reward per episode
- **Collision Rate:** % episodes yang collision

### Efficiency Metrics
- **Q-table Coverage:** % states yang sudah dikunjungi
- **Training Time:** Waktu untuk converge
- **Path Efficiency:** Rasio jarak optimal vs jarak aktual

---

## 🔧 Tuning Tips

### Jika Robot Terlalu Exploratory
- Turunkan `epsilon_decay` (misal: 0.995)
- Turunkan `epsilon_min` (misal: 0.01)

### Jika Robot Stuck di Local Optima
- Naikkan `learning_rate` (misal: 0.3)
- Naikkan `epsilon_min` (misal: 0.1)

### Jika Training Terlalu Lambat
- Kurangi `max_step` (misal: 1000)
- Naikkan `learning_rate` (misal: 0.3)
- Perbesar reward scale

### Jika Robot Terlalu Agresif
- Turunkan `linear_vel` (misal: 0.15)
- Turunkan `angular_vel` values
- Perbesar `collision_threshold` (misal: 0.2)

---

## 📚 References

1. Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
3. ROBOTIS TurtleBot3 Machine Learning: https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning

---

## 📄 License

Apache License 2.0

---

## 👥 Authors

- Implementation: Moch Shultan
- Based on: ROBOTIS TurtleBot3 DQN Framework

---

**Last Updated:** 2024
