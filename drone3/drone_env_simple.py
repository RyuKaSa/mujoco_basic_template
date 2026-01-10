"""
Drone Environment with 6 Cardinal Direction Targets

Targets spawn in one of 6 directions:
- Forward, Backward, Left, Right, Up, Down

All at fixed distance, body-relative (except up/down which are world-relative).
"""

import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def quat_to_euler(quat):
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def quat_rotate(quat, vec):
    w, q = quat[0], quat[1:4]
    t = 2.0 * np.cross(q, vec)
    return vec + w * t + np.cross(q, t)


def quat_rotate_inverse(quat, vec):
    w, q = quat[0], -quat[1:4]
    t = 2.0 * np.cross(q, vec)
    return vec + w * t + np.cross(q, t)


# 6 cardinal directions
CARDINAL_DIRECTIONS = {
    'forward':  np.array([ 1.0,  0.0,  0.0]),
    'backward': np.array([-1.0,  0.0,  0.0]),
    'left':     np.array([ 0.0,  1.0,  0.0]),
    'right':    np.array([ 0.0, -1.0,  0.0]),
    'up':       np.array([ 0.0,  0.0,  1.0]),
    'down':     np.array([ 0.0,  0.0, -1.0]),
}


class CardinalDroneEnv(gym.Env):
    """
    Drone env where targets spawn in cardinal directions.
    
    Observation (13D):
        - Target direction in body frame (3)
        - Target distance (1)
        - Linear velocity in body frame (3)
        - Angular velocity (3)
        - Gravity in body frame (3)
        
    Action (4D): Motor commands normalized [-1, 1]
    """
    
    def __init__(self, model_path="model.xml", mode="forward"):
        """
        Args:
            model_path: Path to MuJoCo model
            mode: "forward" = only forward targets, "cardinal" = all 6 directions
        """
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        
        # Mode determines available directions
        self.mode = mode
        if mode == "forward":
            self.available_directions = ['forward']
        else:
            self.available_directions = list(CARDINAL_DIRECTIONS.keys())
        
        # Target settings
        self.target_distance = 5.0
        self.target_pos = np.array([5.0, 0.0, 1.0])
        self.target_radius = 0.5
        self.current_direction = 'forward'
        
        # Limits
        self.max_steps = 5000
        self.steps = 0
        self.targets_reached = 0
        
        # Spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start altitude - higher if we might go down
        self.data.qpos[2] = 5.0
        self.data.qpos[3] = 1.0
        mujoco.mj_forward(self.model, self.data)
        
        self._spawn_target()
        self.steps = 0
        self.targets_reached = 0
        
        return self._get_obs(), {}
    
    def _spawn_target(self):
        """Spawn target in random available direction."""
        # Pick random direction from available ones
        self.current_direction = self.np_random.choice(self.available_directions)
        direction_body = CARDINAL_DIRECTIONS[self.current_direction]
        
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        
        if self.current_direction in ['up', 'down']:
            # World frame for vertical
            self.target_pos = pos + direction_body * self.target_distance
        else:
            # Body frame for horizontal
            direction_world = quat_rotate(quat, direction_body)
            direction_world[2] = 0  # Keep horizontal
            direction_world = direction_world / (np.linalg.norm(direction_world) + 1e-6)
            self.target_pos = pos + direction_world * self.target_distance
            self.target_pos[2] = pos[2]  # Same altitude
        
        # Clamp altitude
        self.target_pos[2] = np.clip(self.target_pos[2], 0.5, 15.0)

        # update taret visual marker
        self.data.mocap_pos[0] = self.target_pos

    
    def step(self, action):
        # Denormalize: [-1,1] -> [0,12]
        motors = (action + 1.0) * 6.0
        motors = np.clip(motors, 0.0, 12.0)
        
        self.data.ctrl[:4] = motors
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        
        pos = self.data.qpos[:3]
        distance = np.linalg.norm(self.target_pos - pos)
        
        reward = -distance * 0.1
        
        target_reached = distance < self.target_radius
        if target_reached:
            reward = 10.0
            self.targets_reached += 1
            self._spawn_target()
        
        terminated = False
        if pos[2] < 0.1:
            terminated = True
            reward = -10.0
        
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {
            "distance": distance,
            "targets_reached": self.targets_reached,
            "target_reached": target_reached,
            "direction": self.current_direction,
        }
    
    def _get_obs(self):
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel_world = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        to_target = self.target_pos - pos
        distance = np.linalg.norm(to_target)
        to_target_body = quat_rotate_inverse(quat, to_target)
        target_dir = to_target_body / (distance + 1e-6)
        
        vel_body = quat_rotate_inverse(quat, vel_world)
        gravity_body = quat_rotate_inverse(quat, np.array([0, 0, -1]))
        
        return np.concatenate([
            target_dir,
            [distance / 10.0],
            vel_body / 5.0,
            ang_vel / 5.0,
            gravity_body,
        ]).astype(np.float32)


gym.register(id='CardinalDrone-v0', entry_point='drone_env_cardinal:CardinalDroneEnv')