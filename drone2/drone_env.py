"""
Drone Environment - Target Navigation with PID Stabilization Layer

Architecture:
    RL Policy → Velocity Setpoints → PID Controller → Motor Commands → Physics

Observation (12D, body-relative):
- Position error to target (3): body frame
- Linear velocity (3): body frame  
- Angular velocity (3): body frame rates
- Gravity direction (3): body frame (encodes orientation)

Action (3D): [vx, vy, vz] velocity setpoints
- vx: Forward/backward velocity (-2 to 2 m/s)
- vy: Left/right velocity (-2 to 2 m/s)
- vz: Up/down velocity (-1 to 1 m/s)

Yaw is handled AUTOMATICALLY - the PID always keeps the drone facing the target.
This simplifies the RL's job to just "move toward target in 3D space".
"""

import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simple_pid import PID


def quat_to_euler(quat):
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat
    
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


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


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class PIDController:
    """
    PID controller that converts velocity setpoints to motor commands.
    
    Includes automatic heading tracking - the drone always faces the target.
    """
    
    def __init__(self, dt=0.002):
        self.dt = dt
        
        # Inner loop PIDs (tuned values from test_pid_mujoco.py)
        self.pid_alt = PID(21.0, 0.0, 0.05, setpoint=0, output_limits=(-3.0, 3.0))
        self.pid_roll = PID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_pitch = PID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_yaw_rate = PID(0.75, 0.0, 0.4, setpoint=0, output_limits=(-1.0, 1.0))
        
        # Outer loop PIDs (velocity to attitude)
        self.pid_vx = PID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        self.pid_vy = PID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        
        # Auto-heading PID (yaw position control)
        # This PID outputs a desired yaw RATE to track the target heading
        self.pid_yaw_pos = PID(2.0, 0.0, 0.3, setpoint=0, output_limits=(-2.0, 2.0))
        
        # Outer loop runs at lower frequency
        self.outer_counter = 0
        self.outer_divider = 20
        
        # Base hover thrust (tuned for your drone model)
        self.HOVER_THRUST = 4.0712
        
    def reset(self):
        """Reset all PID controllers."""
        for pid in [self.pid_alt, self.pid_roll, self.pid_pitch, 
                    self.pid_yaw_rate, self.pid_vx, self.pid_vy, self.pid_yaw_pos]:
            pid.reset()
        self.outer_counter = 0
        
    def compute(self, pos, quat, vel_world, ang_vel, cmd_vx, cmd_vy, cmd_vz, target_pos):
        """
        Compute motor commands from velocity setpoints with auto-heading.
        
        Args:
            pos: World position [x, y, z]
            quat: Orientation quaternion [w, x, y, z]
            vel_world: World frame velocity [vx, vy, vz]
            ang_vel: Angular velocity [wx, wy, wz]
            cmd_vx: Desired forward velocity (body frame, m/s)
            cmd_vy: Desired lateral velocity (body frame, m/s)
            cmd_vz: Desired vertical velocity (world frame, m/s)
            target_pos: Target position for auto-heading [x, y, z]
            
        Returns:
            Motor commands [motor_fr, motor_fl, motor_rl, motor_rr]
        """
        # Convert world velocity to body frame
        vel_body = quat_rotate_inverse(quat, vel_world)
        euler = quat_to_euler(quat)
        roll, pitch, yaw = euler
        
        # === AUTO-HEADING: Compute desired yaw to face target ===
        to_target = target_pos - pos
        horizontal_dist = np.sqrt(to_target[0]**2 + to_target[1]**2)
        
        if horizontal_dist > 0.5:  # Only track heading if not directly above/below target
            desired_yaw = np.arctan2(to_target[1], to_target[0])
            yaw_error = wrap_angle(desired_yaw - yaw)
            # PID outputs desired yaw rate to correct the heading error
            desired_yaw_rate = self.pid_yaw_pos(-yaw_error)  # Negative because we want to reduce error
        else:
            # Too close horizontally - maintain current heading
            desired_yaw_rate = 0.0
        
        # Set velocity setpoints from RL commands
        self.pid_vx.setpoint = cmd_vx
        self.pid_vy.setpoint = cmd_vy
        self.pid_alt.setpoint = cmd_vz
        self.pid_yaw_rate.setpoint = desired_yaw_rate  # Track the auto-computed yaw rate
        
        # Outer loop: velocity error → desired attitude
        self.outer_counter += 1
        if self.outer_counter >= self.outer_divider:
            self.outer_counter = 0
            # Pitch to control forward velocity
            desired_pitch = self.pid_vx(vel_body[0])
            # Roll to control lateral velocity (negative for correct direction)
            desired_roll = -self.pid_vy(vel_body[1])
            
            self.pid_pitch.setpoint = desired_pitch
            self.pid_roll.setpoint = desired_roll
        
        # Inner loop: attitude stabilization
        alt_adjustment = self.pid_alt(vel_world[2])
        cmd_thrust = self.HOVER_THRUST + alt_adjustment
        
        cmd_roll = -self.pid_roll(roll)
        cmd_pitch = -self.pid_pitch(pitch)
        cmd_yaw_out = -self.pid_yaw_rate(ang_vel[2])
        
        # Motor mixing (X-config quadrotor)
        motor_fr = cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw_out
        motor_fl = cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw_out
        motor_rl = cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw_out
        motor_rr = cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw_out
        
        motors = np.array([motor_fr, motor_fl, motor_rl, motor_rr])
        return np.clip(motors, 0.0, 12.0)


class DroneEnv(gym.Env):
    """
    Drone navigation environment with PID stabilization and auto-heading.
    
    The RL agent outputs velocity setpoints [vx, vy, vz], and the PID controller
    handles all low-level stabilization AND automatically keeps the drone
    facing the target.
    """
    
    # Cardinal directions in body frame
    CARDINAL_DIRECTIONS = {
        'forward': np.array([1.0, 0.0, 0.0]),
        'back': np.array([-1.0, 0.0, 0.0]),
        'left': np.array([0.0, 1.0, 0.0]),
        'right': np.array([0.0, -1.0, 0.0]),
        'up': np.array([0.0, 0.0, 1.0]),
        'down': np.array([0.0, 0.0, -1.0]),
    }
    
    # Combined axis pairs
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
        
        # PID controller for low-level stabilization + auto-heading
        self.pid_controller = PIDController(dt=self.dt)
        
        # Velocity limits for action scaling
        self.max_vx = 2.0   # m/s forward/backward
        self.max_vy = 2.0   # m/s left/right
        self.max_vz = 1.0   # m/s up/down
        
        # Target system
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.difficulty = "easy"
        self.easy_ratio = 1.0
        
        # Distance range for targets
        self.min_target_dist = 15.0
        self.max_target_dist = 30.0
        
        # Dwell time requirement
        self.dwell_time = 0.0
        self.target_reached_dist = 0.5
        self._dwell_timer = 0.0

        # Target timeout
        self.target_timeout = 30.0
        self._target_timer = 0.0
        
        # Hard boundaries
        self.boundary_padding = 3.0
        self.max_tilt = 0.9
        self.max_ang_vel = 8.0
        self._bounds_min = np.array([-100, -100, 0.1])
        self._bounds_max = np.array([100, 100, 50])
        
        # === REWARDS ===
        self.distance_scale = 1.0
        self.alive_bonus = -0.02
        self.proximity_bonus = 0.2
        self.target_reached_bonus = 100.0
        self.crash_penalty = -50.0
        
        # Velocity shaping weights (asymmetric)
        self.forward_reward = 0.5       # Reward for moving toward target
        self.backward_penalty = 1.0     # 2x penalty for moving away
        self.sideways_penalty = 0.3     # Penalty for lateral drift
        self.vertical_reward = 0.5      # Reward for correct vertical movement
        self.vertical_penalty = 1.0     # 2x penalty for wrong vertical direction
        
        # Stagnation penalty
        self.min_required_speed = 0.3   # m/s - must keep moving
        self.stagnation_penalty = 0.5   # Penalty scale for being too slow
        
        # Observation: pos_error(3) + lin_vel(3) + ang_vel(3) + gravity(3) = 12D
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        
        # Action: [vx, vy, vz] normalized to [-1, 1] (yaw is automatic!)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 10000
        self.prev_distance = 0.0
        self.targets_reached = 0
        
        # Observation normalization
        self.pos_scale = 10.0
        self.vel_scale = 5.0
        self.ang_vel_scale = 5.0

    def set_curriculum(self, easy_ratio, min_dist, max_dist):
        """Called by curriculum callback to adjust difficulty."""
        self.easy_ratio = easy_ratio
        self.min_target_dist = min_dist
        self.max_target_dist = max_dist

    def _spawn_target(self):
        """Spawn a new target relative to current drone position."""
        current_pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7]

        self._target_timer = 0.0
        
        is_easy = self.np_random.random() < self.easy_ratio
        dist = self.np_random.uniform(self.min_target_dist, self.max_target_dist)
        
        if is_easy:
            direction_name = self.np_random.choice(list(self.CARDINAL_DIRECTIONS.keys()))
            direction_body = self.CARDINAL_DIRECTIONS[direction_name]
            direction_world = quat_rotate(quat, direction_body)
        else:
            axis1_name, axis2_name = self.COMBINED_AXES[
                self.np_random.integers(len(self.COMBINED_AXES))
            ]
            dir1_body = self.CARDINAL_DIRECTIONS[axis1_name]
            dir2_body = self.CARDINAL_DIRECTIONS[axis2_name]
            
            weight = self.np_random.uniform(0.3, 0.7)
            combined_body = weight * dir1_body + (1 - weight) * dir2_body
            combined_body = combined_body / np.linalg.norm(combined_body)
            direction_world = quat_rotate(quat, combined_body)
        
        new_target = current_pos + direction_world * dist
        new_target[2] = np.clip(new_target[2], 0.5, 20.0)
        
        self.target_pos = new_target
        self.prev_distance = np.linalg.norm(self.target_pos - current_pos)
        self._dwell_timer = 0.0
        
        # Compute bounding box
        padding = self.boundary_padding
        self._bounds_min = np.minimum(current_pos, self.target_pos) - padding
        self._bounds_max = np.maximum(current_pos, self.target_pos) + padding
        self._bounds_min[2] = max(0.15, self._bounds_min[2])
        self._bounds_max[2] = min(30.0, self._bounds_max[2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start position
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 1.0
        self.data.qpos[3] = 1.0  # quat w
        self.data.qpos[4:7] = 0.0
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
        # Reset PID controller
        self.pid_controller.reset()
        
        # Spawn first target
        self._spawn_target()
        
        self.steps = 0
        self.targets_reached = 0
        self._dwell_timer = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # Scale actions to velocity setpoints (3D only - yaw is automatic)
        cmd_vx = action[0] * self.max_vx
        cmd_vy = action[1] * self.max_vy
        cmd_vz = action[2] * self.max_vz
        
        # Get current state
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        # PID controller computes motor commands (with auto-heading toward target)
        motors = self.pid_controller.compute(
            pos, quat, vel, ang_vel,
            cmd_vx, cmd_vy, cmd_vz, self.target_pos  # Pass target for auto-heading
        )
        
        # Apply motor commands
        self.data.ctrl[:4] = motors
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Get new state
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        to_target = self.target_pos - pos
        distance = np.linalg.norm(to_target)
        
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = quat_rotate_inverse(quat, gravity_world)
        tilt = 1.0 - (-gravity_body[2])
        
        ang_vel_mag = np.linalg.norm(ang_vel)
        self._target_timer += self.dt
        
        # === REWARDS ===
        reward = 0.0

        # 1. Distance delta (primary reward signal)
        distance_delta = self.prev_distance - distance
        reward += self.distance_scale * distance_delta

        # 2. Asymmetric velocity shaping (horizontal)
        to_target_horiz = to_target.copy()
        to_target_horiz[2] = 0
        horiz_dist = np.linalg.norm(to_target_horiz)
        
        if horiz_dist > 0.1:
            to_target_horiz_dir = to_target_horiz / horiz_dist
            
            # Forward speed toward target (horizontal only)
            forward_speed_horiz = np.dot(vel[:2], to_target_horiz_dir[:2])
            
            # Sideways speed (perpendicular to target direction)
            vel_forward_horiz = forward_speed_horiz * to_target_horiz_dir[:2]
            sideways_speed = np.linalg.norm(vel[:2] - vel_forward_horiz)
            
            # Reward/penalize horizontal movement
            if forward_speed_horiz >= 0:
                reward += self.forward_reward * forward_speed_horiz
            else:
                reward += self.backward_penalty * forward_speed_horiz  # Already negative
            
            # Penalize sideways drift
            reward -= self.sideways_penalty * sideways_speed

        # 3. Asymmetric velocity shaping (vertical)
        vertical_error = to_target[2]  # Positive = target is above
        vertical_vel = vel[2]          # Positive = moving up
        
        # Check if moving in correct vertical direction
        correct_direction = vertical_vel * vertical_error
        
        if correct_direction >= 0:
            reward += self.vertical_reward * abs(vertical_vel)
        else:
            reward -= self.vertical_penalty * abs(vertical_vel)
        
        # 4. Small time penalty
        reward += self.alive_bonus
        
        # === DWELL & TARGET REACHED ===
        target_reached = False
        in_target_zone = distance < self.target_reached_dist

        # 5. Stagnation penalty (only outside target zone)
        if not in_target_zone:
            speed = np.linalg.norm(vel)
            if speed < self.min_required_speed:
                reward -= self.stagnation_penalty * (self.min_required_speed - speed)

        if in_target_zone:
            self._dwell_timer += self.dt
            reward += self.proximity_bonus
            
            if self._dwell_timer >= self.dwell_time:
                target_reached = True
                reward += self.target_reached_bonus
                self.targets_reached += 1
                print(f"Target reached! Total: {self.targets_reached}")
                self._spawn_target()
                distance = np.linalg.norm(self.target_pos - pos)
        else:
            self._dwell_timer = 0.0
        
        self.prev_distance = distance
        
        # === TERMINATION CONDITIONS ===
        terminated = False
        crash_reason = None
        
        # Out of bounds
        if np.any(pos < self._bounds_min) or np.any(pos > self._bounds_max):
            reward = self.crash_penalty
            terminated = True
            crash_reason = "out_of_bounds"
        
        # Excessive tilt
        elif tilt > self.max_tilt:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_tilt"
        
        # Excessive angular velocity
        elif ang_vel_mag > self.max_ang_vel:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_spin"
        
        # Ground crash
        elif pos[2] < 0.05:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "ground_crash"
        
        # Timeout
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
            pos_error_body,
            lin_vel_body,
            ang_vel_normalized,
            gravity_body,
        ]).astype(np.float32)


gym.register(id='Drone-v0', entry_point='drone_env:DroneEnv', max_episode_steps=100000)


if __name__ == "__main__":
    print("=" * 60)
    print("Drone Environment Test - PID + Auto-Heading")
    print("=" * 60)
    
    env = DroneEnv()
    env.easy_ratio = 1.0
    env.min_target_dist = 5.0
    env.max_target_dist = 10.0
    
    obs, _ = env.reset(seed=42)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Action meaning: [vx, vy, vz] (yaw is automatic!)")
    print(f"Start pos: {env.data.qpos[:3]}")
    print(f"Target: {env.target_pos}")
    
    # Test forward movement with auto-heading
    print("\n--- Testing forward velocity + auto-heading ---")
    obs, _ = env.reset(seed=43)
    
    # Place target to the side to test auto-heading
    env.target_pos = env.data.qpos[:3] + np.array([5.0, 5.0, 0.0])
    env.prev_distance = np.linalg.norm(env.target_pos - env.data.qpos[:3])
    
    print(f"Target at: {env.target_pos}")
    print(f"Initial yaw: {np.degrees(quat_to_euler(env.data.qpos[3:7])[2]):.1f}°")
    
    for i in range(1000):
        # Just command forward - auto-heading should turn us toward target
        action = np.array([1.0, 0.0, 0.0])
        obs, reward, term, trunc, info = env.step(action)
        
        if term:
            print(f"Terminated at step {i}: {info['crash_reason']}")
            break
        
        if i % 200 == 0:
            pos = env.data.qpos[:3]
            yaw = np.degrees(quat_to_euler(env.data.qpos[3:7])[2])
            to_target = env.target_pos - pos
            desired_yaw = np.degrees(np.arctan2(to_target[1], to_target[0]))
            print(f"Step {i}: dist={info['distance']:.1f}m, "
                  f"yaw={yaw:.1f}° (target heading: {desired_yaw:.1f}°)")
    
    # Test hover (zero command)
    print("\n--- Testing hover (zero command) ---")
    obs, _ = env.reset(seed=44)
    start_z = env.data.qpos[2]
    
    for i in range(500):
        action = np.array([0.0, 0.0, 0.0])
        obs, reward, term, trunc, info = env.step(action)
        
        if term:
            print(f"Terminated at step {i}: {info['crash_reason']}")
            break
    
    final_z = env.data.qpos[2]
    print(f"Altitude change: {final_z - start_z:.3f}m (should be ~0)")
    
    # Test stagnation penalty
    print("\n--- Testing stagnation penalty ---")
    obs, _ = env.reset(seed=45)
    
    total_reward = 0
    for i in range(100):
        action = np.array([0.0, 0.0, 0.0])  # Hover in place
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term:
            break
    
    print(f"100 steps of hovering: total reward = {total_reward:.2f}")
    print("(Should be negative due to stagnation + alive penalty)")
    
    env.close()
    print("\nTest complete!")