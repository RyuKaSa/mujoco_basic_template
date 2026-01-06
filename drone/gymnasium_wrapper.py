"""
Simple Drone Environment (Fixed)

Observation (12D, body-relative):
- Position error to target (3): body frame
- Linear velocity (3): body frame  
- Angular velocity (3): body frame rates
- Gravity direction (3): body frame (encodes orientation)

Action (4D): [thrust, pitch, roll, yaw] in [-1, 1]

Reward: distance_delta (closer = positive)
      + alive_bonus (small per-step reward for not crashing)
      - control_cost (penalize large actions)
      - tilt_penalty (penalize excessive tilt)
      + target_bonus (bonus for being close to target)
"""

import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from motor_mixer import mix


def quat_rotate(quat, vec):
    """Rotate vector by quaternion (body to world frame).
    
    Uses the efficient formula: v' = v + 2*w*(q_xyz × v) + 2*(q_xyz × (q_xyz × v))
    Simplified: v' = v + w*t + (q_xyz × t) where t = 2*(q_xyz × v)
    """
    w = quat[0]
    q_xyz = quat[1:4]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


def quat_rotate_inverse(quat, vec):
    """Rotate vector by inverse of quaternion (world to body frame).
    
    Same as quat_rotate but with conjugate (negate xyz components).
    """
    w = quat[0]
    q_xyz = -quat[1:4]  # Conjugate for inverse
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


class DroneEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        
        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path("model.xml")
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        
        # Target in world frame
        self.target_pos = np.array([0.0, 0.0, 1.0])
        
        # Mode settings
        self.idle_ratio = 0.5
        self.target_range = 2.0
        
        # Reward weights
        self.distance_scale = 1.0      # Scale for distance delta reward
        self.alive_bonus = 0.1         # Small bonus for staying alive
        self.control_cost = 0.001       # Penalty for large actions
        self.tilt_penalty = 0.01        # Penalty for excessive tilt
        self.target_bonus_dist = 0.2   # Distance threshold for bonus
        self.target_bonus = 0.5        # Bonus when close to target
        
        # Observation: pos_error(3) + lin_vel(3) + ang_vel(3) + gravity(3) = 12D
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        
        # Action: [thrust, pitch, roll, yaw] in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 100000  # Match gym.register
        self.prev_distance = 0.0
        
        # For observation normalization (approximate scales)
        self.pos_scale = 5.0
        self.vel_scale = 5.0
        self.ang_vel_scale = 5.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start position
        start_z = 1.0
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = start_z
        self.data.qpos[3] = 1.0  # quat w
        self.data.qpos[4:7] = 0.0  # quat xyz
        
        # Explicitly zero velocities
        self.data.qvel[:] = 0.0
        
        # Decide mode: idle or random target
        if self.np_random.random() < self.idle_ratio:
            # Idle: target at current position
            self.target_pos = np.array([0.0, 0.0, start_z])
        else:
            # Random target (reduced range for curriculum)
            r = self.target_range
            self.target_pos = np.array([
                self.np_random.uniform(-r, r),
                self.np_random.uniform(-r, r),
                self.np_random.uniform(0.5, r)
            ])
        
        mujoco.mj_forward(self.model, self.data)
        
        self.steps = 0
        self.prev_distance = np.linalg.norm(self.target_pos - self.data.qpos[:3])
        
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        motors = mix(action)
        self.data.ctrl[:4] = motors
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # Current distance
        distance = np.linalg.norm(self.target_pos - self.data.qpos[:3])
        
        # === Reward components ===
        
        # 1. Distance delta (positive when getting closer)
        distance_delta = self.prev_distance - distance
        reward = self.distance_scale * distance_delta
        
        # 2. Alive bonus
        reward += self.alive_bonus
        
        # 3. Control cost (penalize large actions)
        reward -= self.control_cost * np.sum(action ** 2)
        
        # 4. Tilt penalty (penalize when not upright)
        quat = self.data.qpos[3:7]
        gravity_world = np.array([0, 0, -1])
        gravity_body = quat_rotate_inverse(quat, gravity_world)
        # gravity_body[2] = -1 when upright, approaches 0 or +1 when tilted/flipped
        tilt = 1.0 - (-gravity_body[2])  # 0 when upright, 2 when inverted
        reward -= self.tilt_penalty * tilt
        
        # 5. Target bonus (when close)
        if distance < self.target_bonus_dist:
            reward += self.target_bonus
        
        # Update for next step
        self.prev_distance = distance
        
        # Termination conditions
        terminated = False
        
        # Ground touch
        if self.data.qpos[2] < 0.05:
            reward = -10.0
            terminated = True
        
        # Flipped over (too tilted)
        if gravity_body[2] > 0.5:  # Significantly inverted
            reward = -5.0
            terminated = True
        
        # Too far from target (give up)
        if distance > 20.0:
            reward = -5.0
            terminated = True
        
        self.steps += 1
        truncated = self.steps >= self.max_steps
        
        return obs, reward, terminated, truncated, {
            "distance": distance,
            "tilt": tilt,
        }

    def _get_obs(self):
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        lin_vel_world = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]  # Already in body frame for MuJoCo
        
        # Position error in body frame (normalized)
        pos_error_world = self.target_pos - pos
        pos_error_body = quat_rotate_inverse(quat, pos_error_world) / self.pos_scale
        
        # Linear velocity in body frame (normalized)
        lin_vel_body = quat_rotate_inverse(quat, lin_vel_world) / self.vel_scale
        
        # Angular velocity (normalized)
        ang_vel_normalized = ang_vel / self.ang_vel_scale
        
        # Gravity direction in body frame (encodes orientation)
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = quat_rotate_inverse(quat, gravity_world)
        
        return np.concatenate([
            pos_error_body,      # 3
            lin_vel_body,        # 3
            ang_vel_normalized,  # 3
            gravity_body,        # 3
        ]).astype(np.float32)


gym.register(id='Drone-v0', entry_point='drone_env_fixed:DroneEnv', max_episode_steps=100000)


if __name__ == "__main__":
    env = DroneEnv()
    obs, _ = env.reset(seed=42)
    print(f"Obs shape: {obs.shape}")
    print(f"Obs: {obs}")
    print(f"Target: {env.target_pos}")
    
    # Test with neutral action (should hover roughly)
    total_reward = 0
    for i in range(200):
        obs, r, term, trunc, info = env.step(np.array([0, 0, 0, 0]))
        total_reward += r
        if term:
            print(f"Terminated at step {i}")
            break
    
    print(f"Final distance: {info['distance']:.3f}")
    print(f"Final tilt: {info['tilt']:.3f}")
    print(f"Total reward: {total_reward:.2f}")
    env.close()