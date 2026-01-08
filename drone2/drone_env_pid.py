"""
Drone Environment with PID Layer - Target Navigation

Architecture:
    [ Motors ]
         ↑
    [ Mixer (hard constraints, algebraic) ]
         ↑
    [ Inner-loop PID (rates, attitude, altitude) ]
         ↑
    [ RL Policy (velocity commands) ]

RL Action Space (4D): [cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate] in [-1, 1]
    - Scaled to actual velocity/rate commands
    - PID layer handles attitude stabilization

The PID layer provides:
    - Attitude stabilization (roll, pitch from velocity errors)
    - Altitude rate control
    - Yaw rate control
"""

import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simple_pid import PID

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


def quat_to_euler(quat):
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]."""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


class PIDController:
    """
    Wrapper for PID controllers that handle the inner and outer loops.
    
    Architecture:
        [RL velocity commands] -> [Outer PIDs] -> [attitude setpoints]
                                                        |
                                               [Inner PIDs] -> [mixer commands]
    
    Based on the working reference PID code.
    """
    
    def __init__(self, dt: float = 0.002):
        self.dt = dt
        
        # === INNER LOOP PIDs (attitude stabilization) ===
        # From reference: pid_alt = PID(5.50844, 0.57871, 1.2)
        # BUT: reference controls altitude directly, we control vertical velocity
        self.pid_alt = PID(
            Kp=5.5, Ki=0.5, Kd=1.2,
            setpoint=0,
            sample_time=dt,
            output_limits=(-2.5, 2.5)
        )
        
        # From reference: pid_roll = PID(2.6785, 0.56871, 1.2508, limits=(-1,1))
        self.pid_roll = PID(
            Kp=2.7, Ki=0.5, Kd=1.25,
            setpoint=0,
            sample_time=dt,
            output_limits=(-1.0, 1.0)
        )
        
        # From reference: pid_pitch = PID(2.6785, 0.56871, 1.2508, limits=(-1,1))
        self.pid_pitch = PID(
            Kp=2.7, Ki=0.5, Kd=1.25,
            setpoint=0,
            sample_time=dt,
            output_limits=(-1.0, 1.0)
        )
        
        # From reference: pid_yaw = PID(0.54, 0, 5.358333, limits=(-3,3))
        # Note: reference has very high Kd, we keep it moderate
        self.pid_yaw = PID(
            Kp=0.5, Ki=0.0, Kd=1.0,
            setpoint=0,
            sample_time=dt,
            output_limits=(-1.0, 1.0)
        )
        
        # === OUTER LOOP PIDs (velocity tracking) ===
        # From reference: pid_v_x = PID(0.1, 0.003, 0.02, limits=(-0.1, 0.1))
        # We use slightly higher gains and limits for more responsive control
        self.pid_vx = PID(
            Kp=0.12, Ki=0.005, Kd=0.03,
            setpoint=0,
            sample_time=dt,
            output_limits=(-0.15, 0.15)  # Max ~8.5 degrees pitch
        )
        
        # From reference: pid_v_y = PID(0.1, 0.003, 0.02, limits=(-0.1, 0.1))
        self.pid_vy = PID(
            Kp=0.12, Ki=0.005, Kd=0.03,
            setpoint=0,
            sample_time=dt,
            output_limits=(-0.15, 0.15)  # Max ~8.5 degrees roll
        )
        
        # Outer loop update counter (runs slower than inner loop)
        # From reference: outer loop runs every 20 steps
        self.outer_loop_counter = 0
        self.outer_loop_divider = 20
        
    def reset(self):
        """Reset all PID controllers (call on episode reset)."""
        for pid in [self.pid_alt, self.pid_roll, self.pid_pitch, 
                    self.pid_yaw, self.pid_vx, self.pid_vy]:
            pid.reset()
        self.outer_loop_counter = 0
        
    def set_velocity_commands(self, cmd_vx: float, cmd_vy: float, 
                               cmd_vz: float, cmd_yaw_rate: float):
        """
        Set the velocity commands from the RL policy.
        
        Args:
            cmd_vx: Desired velocity in body X direction (forward)
            cmd_vy: Desired velocity in body Y direction (left)
            cmd_vz: Desired vertical velocity (up)
            cmd_yaw_rate: Desired yaw rotation rate
        """
        self.pid_vx.setpoint = cmd_vx
        self.pid_vy.setpoint = cmd_vy
        self.pid_alt.setpoint = cmd_vz  # Use as vertical velocity target
        self.pid_yaw.setpoint = cmd_yaw_rate
        
    def compute(self, pos: np.ndarray, quat: np.ndarray, 
                vel_world: np.ndarray, ang_vel: np.ndarray) -> np.ndarray:
        """
        Compute motor commands from current state.
        
        Args:
            pos: Position [x, y, z] in world frame
            quat: Quaternion [w, x, y, z]
            vel_world: Linear velocity in world frame
            ang_vel: Angular velocity [roll_rate, pitch_rate, yaw_rate]
            
        Returns:
            Motor commands [thrust, pitch, roll, yaw] for the mixer
        """
        # Convert velocities to body frame
        vel_body = quat_rotate_inverse(quat, vel_world)
        
        # Get euler angles
        euler = quat_to_euler(quat)  # [roll, pitch, yaw]
        roll, pitch, yaw = euler
        
        # === OUTER LOOP (velocity -> attitude setpoints) ===
        self.outer_loop_counter += 1
        if self.outer_loop_counter >= self.outer_loop_divider:
            self.outer_loop_counter = 0
            
            # Velocity error -> desired pitch (for forward/backward motion)
            # Positive vx error (moving slower than desired) -> pitch forward (negative pitch)
            desired_pitch = -self.pid_vx(vel_body[0])
            
            # Velocity error -> desired roll (for left/right motion)
            # Positive vy error (moving slower than desired left) -> roll right (positive roll)
            desired_roll = self.pid_vy(vel_body[1])
            
            # Update inner loop setpoints
            self.pid_pitch.setpoint = desired_pitch
            self.pid_roll.setpoint = desired_roll
        
        # === INNER LOOP (attitude stabilization) ===
        
        # Altitude/vertical velocity control
        # Using vertical velocity as the controlled variable
        cmd_thrust = self.pid_alt(vel_world[2])
        
        # Attitude control
        cmd_roll = -self.pid_roll(roll)
        cmd_pitch = self.pid_pitch(pitch)
        cmd_yaw = -self.pid_yaw(ang_vel[2])  # Control yaw rate directly
        
        return np.array([cmd_thrust, cmd_pitch, cmd_roll, cmd_yaw])


class DroneEnvPID(gym.Env):
    """
    Drone environment with PID stabilization layer.
    
    The RL policy outputs high-level velocity commands,
    and the PID layer handles low-level attitude control.
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
        
        # === PID CONTROLLER ===
        self.pid_controller = PIDController(dt=self.dt)
        
        # === COMMAND SCALING ===
        # Scale RL actions [-1, 1] to physical units
        self.max_vel_xy = 3.0      # m/s max horizontal velocity command
        self.max_vel_z = 2.0       # m/s max vertical velocity command
        self.max_yaw_rate = 1.5    # rad/s max yaw rate command
        
        # Target system
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.difficulty = "easy"
        self.easy_ratio = 1.0
        
        self.min_target_dist = 15.0
        self.max_target_dist = 30.0
        
        # Dwell time requirement
        self.dwell_time = 3.0
        self.target_reached_dist = 0.5
        self._dwell_timer = 0.0
        
        # Target timeout
        self.target_timeout = 15.0  # Increased for PID-based control
        self._target_timer = 0.0
        
        # === HARD BOUNDARIES ===
        self.boundary_padding = 2.0  # Slightly more forgiving with PID
        self.max_tilt = 0.8
        self.max_ang_vel = 6.0
        self._bounds_min = np.array([-100, -100, 0.1])
        self._bounds_max = np.array([100, 100, 50])
        
        # === REWARDS ===
        self.distance_scale = 1.0
        self.alive_bonus = -0.05
        self.tilt_penalty = 0.05
        self.proximity_bonus = 0.1
        self.target_reached_bonus = 100.0
        self.crash_penalty = -50.0
        
        # === SPACES ===
        # Observation: pos_error(3) + lin_vel(3) + ang_vel(3) + gravity(3) = 12D
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        
        # Action: [cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate] in [-1, 1]
        # These are HIGH-LEVEL commands that the PID layer interprets
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        
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
        new_target[2] = np.clip(new_target[2], 0.3, 20.0)
        
        self.target_pos = new_target
        self.prev_distance = np.linalg.norm(self.target_pos - current_pos)
        self._dwell_timer = 0.0
        
        # Bounding box
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
        
        # Reset PID controllers
        self.pid_controller.reset()
        
        # Spawn first target
        self._spawn_target()
        
        self.steps = 0
        self.targets_reached = 0
        self._dwell_timer = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # === SCALE RL ACTIONS TO PHYSICAL COMMANDS ===
        cmd_vx = action[0] * self.max_vel_xy
        cmd_vy = action[1] * self.max_vel_xy
        cmd_vz = action[2] * self.max_vel_z
        cmd_yaw_rate = action[3] * self.max_yaw_rate
        
        # Set velocity commands for PID controller
        self.pid_controller.set_velocity_commands(cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate)
        
        # Get current state
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel_world = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        # === PID LAYER: Compute stabilized commands ===
        pid_commands = self.pid_controller.compute(pos, quat, vel_world, ang_vel)
        
        # === MIXER: Convert to motor thrusts ===
        motors = mix(pid_commands)
        self.data.ctrl[:4] = motors
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # === COMPUTE REWARDS AND TERMINATION ===
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        pos_error_world = self.target_pos - pos
        distance = np.linalg.norm(pos_error_world)
        
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body = quat_rotate_inverse(quat, gravity_world)
        tilt = 1.0 - (-gravity_body[2])
        
        ang_vel_mag = np.linalg.norm(ang_vel)
        self._target_timer += self.dt
        
        # === REWARDS ===
        reward = 0.0
        
        # Distance delta
        distance_delta = self.prev_distance - distance
        reward += self.distance_scale * distance_delta
        
        # Velocity shaping (horizontal)
        to_target = self.target_pos - pos
        to_target_horiz = to_target.copy()
        to_target_horiz[2] = 0
        horiz_dist = np.linalg.norm(to_target_horiz)
        to_target_horiz_dir = to_target_horiz / (horiz_dist + 1e-6)
        
        forward_speed_horiz = np.dot(vel[:2], to_target_horiz_dir[:2])
        vel_forward_horiz = forward_speed_horiz * to_target_horiz_dir[:2]
        sideways_speed = np.linalg.norm(vel[:2] - vel_forward_horiz)
        
        if forward_speed_horiz >= 0:
            reward += 0.5 * forward_speed_horiz
        else:
            reward += 1.0 * forward_speed_horiz
        reward -= 0.3 * sideways_speed
        
        # Vertical shaping
        vertical_error = to_target[2]
        vertical_vel = vel[2]
        correct_direction = vertical_vel * vertical_error
        if correct_direction >= 0:
            reward += 0.5 * abs(vertical_vel)
        else:
            reward -= 1.0 * abs(vertical_vel)
        
        reward += self.alive_bonus
        reward -= self.tilt_penalty * tilt
        
        # Heading alignment
        forward_world = quat_rotate(quat, np.array([1.0, 0.0, 0.0]))
        to_target_norm = to_target_horiz / (np.linalg.norm(to_target_horiz) + 1e-6)
        heading_alignment = np.dot(forward_world[:2], to_target_norm[:2])
        if heading_alignment >= 0:
            reward += 0.3 * heading_alignment
        else:
            reward += 2.0 * heading_alignment
        
        # === DWELL & TARGET REACHED ===
        target_reached = False
        in_target_zone = distance < self.target_reached_dist
        
        if not in_target_zone:
            speed = np.linalg.norm(vel)
            min_required_speed = 0.3
            if speed < min_required_speed:
                reward -= 0.5 * (min_required_speed - speed)
        
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
        
        # === HARD BOUNDARY CHECKS ===
        terminated = False
        crash_reason = None
        
        if np.any(pos < self._bounds_min) or np.any(pos > self._bounds_max):
            reward = self.crash_penalty
            terminated = True
            crash_reason = "out_of_bounds"
        elif tilt > self.max_tilt:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_tilt"
        elif ang_vel_mag > self.max_ang_vel:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_spin"
        elif abs(ang_vel[2]) > 3.0:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "excessive_yaw"
        elif pos[2] < 0.05:
            reward = self.crash_penalty
            terminated = True
            crash_reason = "ground_crash"
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
            "pid_commands": pid_commands.tolist(),  # Debug info
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


# Register the environment
gym.register(id='DronePID-v0', entry_point='drone_env_pid:DroneEnvPID', max_episode_steps=100000)


if __name__ == "__main__":
    print("=" * 60)
    print("Drone Environment with PID Layer - Test")
    print("=" * 60)
    
    env = DroneEnvPID()
    env.easy_ratio = 1.0
    env.min_target_dist = 5.0  # Start closer for testing
    env.max_target_dist = 10.0
    
    obs, _ = env.reset(seed=42)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Start pos: {env.data.qpos[:3]}")
    print(f"Target: {env.target_pos}")
    
    # Test: hover in place (zero velocity commands)
    print("\n--- Test: Hover (zero commands) ---")
    obs, _ = env.reset(seed=42)
    for i in range(500):
        action = np.array([0.0, 0.0, 0.0, 0.0])  # No velocity commands
        obs, reward, term, trunc, info = env.step(action)
        if term:
            print(f"Crashed at step {i}: {info['crash_reason']}")
            break
        if i % 100 == 0:
            pos = env.data.qpos[:3]
            print(f"Step {i}: pos={pos}, tilt={info['tilt']:.3f}")
    else:
        print(f"Hover test passed! Final pos: {env.data.qpos[:3]}")
    
    # Test: move forward
    print("\n--- Test: Forward velocity command ---")
    obs, _ = env.reset(seed=43)
    for i in range(500):
        action = np.array([0.5, 0.0, 0.0, 0.0])  # Forward velocity
        obs, reward, term, trunc, info = env.step(action)
        if term:
            print(f"Crashed at step {i}: {info['crash_reason']}")
            break
        if i % 100 == 0:
            pos = env.data.qpos[:3]
            vel = env.data.qvel[:3]
            print(f"Step {i}: pos={pos}, vel={vel}")
    else:
        print(f"Forward test complete! Final pos: {env.data.qpos[:3]}")
    
    # Test: climb
    print("\n--- Test: Vertical climb command ---")
    obs, _ = env.reset(seed=44)
    for i in range(500):
        action = np.array([0.0, 0.0, 0.5, 0.0])  # Climb
        obs, reward, term, trunc, info = env.step(action)
        if term:
            print(f"Crashed at step {i}: {info['crash_reason']}")
            break
        if i % 100 == 0:
            pos = env.data.qpos[:3]
            print(f"Step {i}: altitude={pos[2]:.2f}")
    else:
        print(f"Climb test complete! Final altitude: {env.data.qpos[2]:.2f}")
    
    env.close()
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)