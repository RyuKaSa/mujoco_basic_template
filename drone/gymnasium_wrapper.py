"""
Drone Environment - Target Navigation (Simplified with Hard Boundaries)

Observation (12D, body-relative):
- Position error to target (3): body frame
- Linear velocity (3): body frame  
- Angular velocity (3): body frame rates
- Gravity direction (3): body frame (encodes orientation)

Action (4D): [thrust, pitch, roll, yaw] in [-1, 1]

Key Design: Hard boundaries instead of soft penalties
- Bounding box around start→target path kills drone if crossed
- Excessive tilt kills drone
- Excessive angular velocity kills drone
- Simple rewards: distance delta + velocity shaping + dwell
"""

import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from motor_mixer import mix


def quat_rotate(quat, vec):
    """Rotate vector by quaternion (body to world frame)."""
    w = quat[0]
    q_xyz = quat[1:4]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


def quat_rotate_inverse(quat, vec):
    """Rotate vector by inverse of quaternion (world to body frame)."""
    w = quat[0]
    q_xyz = -quat[1:4]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


class DroneEnv(gym.Env):
    
    # Cardinal directions in body frame
    CARDINAL_DIRECTIONS = {
        'forward': np.array([1.0, 0.0, 0.0]),
        'back': np.array([-1.0, 0.0, 0.0]),
        'left': np.array([0.0, 1.0, 0.0]),
        'right': np.array([0.0, -1.0, 0.0]),
        'up': np.array([0.0, 0.0, 1.0]),
        'down': np.array([0.0, 0.0, -1.0]),
    }
    
    # Combined axis pairs (horizontal + vertical combinations)
    COMBINED_AXES = [
        ('forward', 'up'), ('forward', 'down'),
        ('back', 'up'), ('back', 'down'),
        ('left', 'up'), ('left', 'down'),
        ('right', 'up'), ('right', 'down'),
        ('forward', 'left'), ('forward', 'right'),
        ('back', 'left'), ('back', 'right'),
    ]
    
    def __init__(self):
        super().__init__()
        
        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path("model.xml")
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        
        # Target system
        self.target_pos = np.array([0.0, 0.0, 1.0])  # World frame
        self.difficulty = "easy"  # "easy" = cardinal, "hard" = combined
        self.easy_ratio = 1.0  # Probability of easy target (for curriculum)
        
        # Distance range for targets
        self.min_target_dist = 15.0
        self.max_target_dist = 30.0
        
        # Dwell time requirement
        self.dwell_time = 3.0  # Seconds to stay at target
        self.target_reached_dist = 0.5  # Distance threshold
        self._dwell_timer = 0.0  # Current dwell accumulator

        # Target timeout
        self.target_timeout = 10.0  # Seconds to reach target before death
        self._target_timer = 0.0
        
        # === HARD BOUNDARIES (kills drone) ===
        self.boundary_padding = 1.5  # Meters of allowed deviation from path
        self.max_tilt = 0.9  # Kill if tilt exceeds this (0=upright, 2=inverted)
        self.max_ang_vel = 8.0  # Kill if spinning faster than this (rad/s)
        self._bounds_min = np.array([-100, -100, 0.1])
        self._bounds_max = np.array([100, 100, 50])
        
        # === SIMPLE REWARDS ===
        self.distance_scale = 1.0  # Reward for getting closer
        self.alive_bonus = -0.05  # Small penalty per step (encourages speed)
        self.tilt_penalty = 0.05  # Soft penalty for tilt (in addition to hard limit)
        self.proximity_bonus = 0.1  # Bonus while in target zone
        self.target_reached_bonus = 100.0  # Big bonus for completing dwell
        
        # Terminal penalty (same for all crash types)
        self.crash_penalty = -50.0
        
        # Observation: pos_error(3) + lin_vel(3) + ang_vel(3) + gravity(3) = 12D
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        
        # Action: [thrust, pitch, roll, yaw] in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 10000
        self.prev_distance = 0.0
        self.targets_reached = 0
        
        # For observation normalization
        self.pos_scale = 10.0
        self.vel_scale = 5.0
        self.ang_vel_scale = 5.0

    def set_curriculum(self, easy_ratio, min_dist, max_dist):
        """Called by curriculum callback to adjust difficulty."""
        self.easy_ratio = easy_ratio
        self.min_target_dist = min_dist
        self.max_target_dist = max_dist

    def _get_direction_vector(self, direction_name):
        """Get world-frame direction based on drone's current orientation."""
        quat = self.data.qpos[3:7]
        body_dir = self.CARDINAL_DIRECTIONS[direction_name]
        return quat_rotate(quat, body_dir)

    def _spawn_target(self):
        """Spawn a new target relative to current drone position."""
        current_pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7]

        self._target_timer = 0.0
        
        # Decide difficulty for this target
        is_easy = self.np_random.random() < self.easy_ratio
        
        # Random distance
        dist = self.np_random.uniform(self.min_target_dist, self.max_target_dist)
        
        if is_easy:
            # Cardinal direction (one axis only)
            direction_name = self.np_random.choice(list(self.CARDINAL_DIRECTIONS.keys()))
            direction_body = self.CARDINAL_DIRECTIONS[direction_name]
            direction_world = quat_rotate(quat, direction_body)
        else:
            # Combined axes (two axes)
            axis1_name, axis2_name = self.COMBINED_AXES[
                self.np_random.integers(len(self.COMBINED_AXES))
            ]
            dir1_body = self.CARDINAL_DIRECTIONS[axis1_name]
            dir2_body = self.CARDINAL_DIRECTIONS[axis2_name]
            
            # Random weighting between the two axes
            weight = self.np_random.uniform(0.3, 0.7)
            combined_body = weight * dir1_body + (1 - weight) * dir2_body
            combined_body = combined_body / np.linalg.norm(combined_body)
            direction_world = quat_rotate(quat, combined_body)
        
        # Compute target position
        new_target = current_pos + direction_world * dist
        
        # Clamp height to valid range
        new_target[2] = np.clip(new_target[2], 0.3, 20.0)
        
        self.target_pos = new_target
        self.prev_distance = np.linalg.norm(self.target_pos - current_pos)
        self._dwell_timer = 0.0
        
        # === COMPUTE BOUNDING BOX ===
        padding = self.boundary_padding
        self._bounds_min = np.minimum(current_pos, self.target_pos) - padding
        self._bounds_max = np.maximum(current_pos, self.target_pos) + padding
        
        # Ensure reasonable height bounds
        self._bounds_min[2] = max(0.15, self._bounds_min[2])
        self._bounds_max[2] = min(30.0, self._bounds_max[2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start position
        start_z = 1.0
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = start_z
        self.data.qpos[3] = 1.0  # quat w
        self.data.qpos[4:7] = 0.0
        
        # Zero velocities
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
        # Spawn first target
        self._spawn_target()
        
        self.steps = 0
        self.targets_reached = 0
        self._dwell_timer = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        motors = mix(action)
        self.data.ctrl[:4] = motors
        
        mujoco.mj_step(self.model, self.data)
        
        # Get state
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        pos_error_world = self.target_pos - pos
        distance = np.linalg.norm(pos_error_world)
        
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = quat_rotate_inverse(quat, gravity_world)
        tilt = 1.0 - (-gravity_body[2])  # 0 = upright, 2 = inverted
        
        ang_vel_mag = np.linalg.norm(ang_vel)

        self._target_timer += self.dt
        
        # === REWARDS ===
        reward = 0.0

        # 0. Distance delta (core gradient)
        distance_delta = self.prev_distance - distance
        reward += self.distance_scale * distance_delta

        # 1. Velocity-based rewards (separate horizontal & vertical)
        to_target = self.target_pos - pos

        # --- HORIZONTAL PLANE ---
        to_target_horiz = to_target.copy()
        to_target_horiz[2] = 0
        horiz_dist = np.linalg.norm(to_target_horiz)
        to_target_horiz_dir = to_target_horiz / (horiz_dist + 1e-6)

        # Horizontal velocity components
        forward_speed_horiz = np.dot(vel[:2], to_target_horiz_dir[:2])

        # Sideways in horizontal plane only
        vel_forward_horiz = forward_speed_horiz * to_target_horiz_dir[:2]
        sideways_speed = np.linalg.norm(vel[:2] - vel_forward_horiz)

        # Reward/punish horizontal movement
        if forward_speed_horiz >= 0:
            reward += 0.5 * forward_speed_horiz
        else:
            reward += 1.0 * forward_speed_horiz  # 2x penalty for backing up

        reward -= 0.3 * sideways_speed

        # --- VERTICAL AXIS ---
        vertical_error = to_target[2]  # Positive = target is above
        vertical_vel = vel[2]          # Positive = moving up

        # Reward moving toward target's altitude, punish moving away
        correct_direction = vertical_vel * vertical_error

        if correct_direction >= 0:
            reward += 0.5 * abs(vertical_vel)   # Correct vertical direction
        else:
            reward -= 1.0 * abs(vertical_vel)   # Wrong vertical direction (2x penalty)
        
        # 2. Alive bonus (negative = time pressure)
        reward += self.alive_bonus
        
        # 3. Soft tilt penalty
        reward -= self.tilt_penalty * tilt

        # 4. Heading alignment reward
        forward_world = quat_rotate(quat, np.array([1.0, 0.0, 0.0]))
        to_target_horiz_for_heading = to_target.copy()
        to_target_horiz_for_heading[2] = 0
        to_target_norm = to_target_horiz_for_heading / (np.linalg.norm(to_target_horiz_for_heading) + 1e-6)
        heading_alignment = np.dot(forward_world[:2], to_target_norm[:2])

        if heading_alignment >= 0:
            reward += 0.3 * heading_alignment
        else:
            reward += 2.0 * heading_alignment  # Stronger penalty when facing wrong way

        # === DWELL & TARGET REACHED ===
        target_reached = False
        in_target_zone = distance < self.target_reached_dist
    
        # === STAGNATION PENALTY (only outside target zone) ===
        if not in_target_zone:
            speed = np.linalg.norm(vel)
            min_required_speed = 0.3  # m/s - must keep moving toward target
            
            if speed < min_required_speed:
                reward -= 0.5 * (min_required_speed - speed)

        if in_target_zone:
            # Accumulate dwell time
            self._dwell_timer += self.dt
            
            # Proximity bonus while dwelling
            reward += self.proximity_bonus
            
            # Check if dwell complete
            if self._dwell_timer >= self.dwell_time:
                target_reached = True
                reward += self.target_reached_bonus
                
                self.targets_reached += 1
                print(f"Target reached! Total: {self.targets_reached}")
                self._spawn_target()
                distance = np.linalg.norm(self.target_pos - pos)
        else:
            # Reset dwell timer if outside zone
            self._dwell_timer = 0.0
        
        self.prev_distance = distance
        
        # === HARD BOUNDARY CHECKS (instant death) ===
        
        terminated = False
        crash_reason = None
        
        # 1. Out of bounding box
        if np.any(pos < self._bounds_min) or np.any(pos > self._bounds_max):
            reward = self.crash_penalty
            terminated = True
            crash_reason = "out_of_bounds"
        
        # 2. Excessive tilt
        elif tilt > self.max_tilt:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_tilt"
        
        # 3. Excessive angular velocity (spinning out of control)
        elif ang_vel_mag > self.max_ang_vel:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_spin"

        # 4. Excessive yaw rate
        yaw_rate = abs(ang_vel[2])
        if yaw_rate > 3.0:  # rad/s, ~170 deg/s
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_yaw"
        
        # 5. Ground crash
        elif pos[2] < 0.05:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "ground_crash"
        
        # 6. Timeout
        elif self._target_timer >= self.target_timeout:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "timeout"
        
        self.steps += 1
        truncated = self.steps >= self.max_steps
        
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {
            "distance": distance,
            "tilt": tilt,
            "ang_vel": ang_vel_mag,
            "targets_reached": self.targets_reached,
            "target_reached_this_step": target_reached,
            "dwell_progress": self._dwell_timer / self.dwell_time if self.dwell_time > 0 else 0,
            "in_target_zone": in_target_zone,
            "crash_reason": crash_reason,
        }

    def _get_obs(self):
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        lin_vel_world = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        pos_error_world = self.target_pos - pos
        pos_error_body = quat_rotate_inverse(quat, pos_error_world) / self.pos_scale
        lin_vel_body = quat_rotate_inverse(quat, lin_vel_world) / self.vel_scale
        ang_vel_normalized = ang_vel / self.ang_vel_scale
        
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = quat_rotate_inverse(quat, gravity_world)
        
        return np.concatenate([
            pos_error_body,       # 3D - where is target
            lin_vel_body,         # 3D - how am I moving
            ang_vel_normalized,   # 3D - how am I rotating
            gravity_body,         # 3D - which way is up
        ]).astype(np.float32)


gym.register(id='Drone-v0', entry_point='drone_env:DroneEnv', max_episode_steps=100000)


if __name__ == "__main__":
    print("=" * 60)
    print("Drone Environment Test - Hard Boundaries")
    print("=" * 60)
    
    env = DroneEnv()
    env.easy_ratio = 1.0
    env.min_target_dist = 15.0
    env.max_target_dist = 30.0
    env.boundary_padding = 1.0
    
    obs, _ = env.reset(seed=42)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Start pos: {env.data.qpos[:3]}")
    print(f"Target: {env.target_pos}")
    print(f"Bounds min: {env._bounds_min}")
    print(f"Bounds max: {env._bounds_max}")
    
    # Test boundary violation
    print("\n--- Testing boundary crash ---")
    env.data.qpos[0] = env._bounds_max[0] + 0.1
    mujoco.mj_forward(env.model, env.data)
    obs, reward, term, trunc, info = env.step(np.array([0, 0, 0, 0]))
    print(f"Moved outside X bound")
    print(f"Terminated: {term}, Reason: {info['crash_reason']}, Reward: {reward}")
    
    # Test tilt crash
    print("\n--- Testing tilt crash ---")
    obs, _ = env.reset(seed=43)
    env.data.qpos[3:7] = [0.7, 0.7, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    obs, reward, term, trunc, info = env.step(np.array([0, 0, 0, 0]))
    print(f"Tilted drone")
    print(f"Tilt: {info['tilt']:.2f}, Terminated: {term}, Reason: {info['crash_reason']}")
    
    # Test spin crash
    print("\n--- Testing spin crash ---")
    obs, _ = env.reset(seed=44)
    env.data.qvel[3:6] = [10, 0, 0]
    obs, reward, term, trunc, info = env.step(np.array([0, 0, 0, 0]))
    print(f"Spinning drone")
    print(f"Ang vel: {info['ang_vel']:.2f}, Terminated: {term}, Reason: {info['crash_reason']}")
    
    # Normal flight test
    print("\n--- Normal flight test ---")
    obs, _ = env.reset(seed=45)
    for i in range(100):
        action = np.array([0.1, 0, 0, 0])
        obs, reward, term, trunc, info = env.step(action)
        if term:
            print(f"Crashed at step {i}: {info['crash_reason']}")
            break
    else:
        print(f"Survived 100 steps, distance: {info['distance']:.2f}m")
    
    env.close()