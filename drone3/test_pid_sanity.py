"""
Debug PID - See what's actually happening
"""

import numpy as np
from simple_pid import PID
import mujoco
import mujoco.viewer
import time

from drone_env_simple import SimpleDroneEnv, quat_to_euler, quat_rotate_inverse


class PIDAutopilot:
    def __init__(self):
        self.pid_alt = PID(21.0, 0.0, 0.05, setpoint=0, output_limits=(-3.0, 3.0))
        self.pid_roll = PID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_pitch = PID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_yaw = PID(0.75, 0.0, 0.4, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_vx = PID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        self.pid_vy = PID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        
        self.outer_counter = 0
        self.outer_divider = 20
        self.HOVER_THRUST = 4.0712
        
    def reset(self):
        for pid in [self.pid_alt, self.pid_roll, self.pid_pitch, 
                    self.pid_yaw, self.pid_vx, self.pid_vy]:
            pid.reset()
        self.outer_counter = 0
    
    def compute(self, pos, quat, vel_world, ang_vel, target_pos, debug=False):
        vel_body = quat_rotate_inverse(quat, vel_world)
        euler = quat_to_euler(quat)
        roll, pitch, yaw = euler
        
        # Direction to target in body frame
        to_target = target_pos - pos
        to_target_body = quat_rotate_inverse(quat, to_target)
        
        # Desired velocities (body frame for vx/vy, world frame for vz)
        cmd_vx = np.clip(to_target_body[0] * 0.5, -2.0, 2.0)
        cmd_vy = np.clip(to_target_body[1] * 0.5, -2.0, 2.0)
        cmd_vz = np.clip(to_target[2] * 0.5, -1.0, 1.0)
        
        # Set velocity setpoints
        self.pid_vx.setpoint = cmd_vx
        self.pid_vy.setpoint = cmd_vy
        self.pid_alt.setpoint = cmd_vz
        self.pid_yaw.setpoint = 0  # No yaw command for now
        
        # Outer loop: velocity -> attitude
        self.outer_counter += 1
        if self.outer_counter >= self.outer_divider:
            self.outer_counter = 0
            desired_pitch = self.pid_vx(vel_body[0])
            desired_roll = -self.pid_vy(vel_body[1])
            self.pid_pitch.setpoint = desired_pitch
            self.pid_roll.setpoint = desired_roll
        
        # Inner loop: attitude stabilization
        alt_adj = self.pid_alt(vel_world[2])
        cmd_thrust = self.HOVER_THRUST + alt_adj
        
        cmd_roll = -self.pid_roll(roll)
        cmd_pitch = -self.pid_pitch(pitch)
        cmd_yaw_out = -self.pid_yaw(ang_vel[2])
        
        # Motor mixing
        motor_fr = cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw_out
        motor_fl = cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw_out
        motor_rl = cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw_out
        motor_rr = cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw_out
        
        motors = np.clip([motor_fr, motor_fl, motor_rl, motor_rr], 0.0, 12.0)
        
        if debug:
            print(f"  pos={pos}, alt={pos[2]:.2f}")
            print(f"  euler=({np.degrees(roll):.1f}, {np.degrees(pitch):.1f}, {np.degrees(yaw):.1f})")
            print(f"  vel_world={vel_world}, vel_body={vel_body}")
            print(f"  to_target_body={to_target_body}")
            print(f"  cmd_vel=({cmd_vx:.2f}, {cmd_vy:.2f}, {cmd_vz:.2f})")
            print(f"  thrust={cmd_thrust:.2f}, alt_adj={alt_adj:.2f}")
            print(f"  motors={motors}")
            print()
        
        return motors


def debug_run():
    print("=" * 60)
    print("DEBUG PID RUN")
    print("=" * 60)
    
    env = SimpleDroneEnv()
    env.target_distance = 3.0
    
    # Manual reset
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[2] = 1.0  # Start at 1m
    env.data.qpos[3] = 1.0  # quat w = 1
    mujoco.mj_forward(env.model, env.data)
    
    # Set a simple target: 3m ahead
    env.target_pos = np.array([3.0, 0.0, 1.0])
    
    pid = PIDAutopilot()
    pid.reset()
    
    print(f"Initial state:")
    print(f"  pos: {env.data.qpos[:3]}")
    print(f"  quat: {env.data.qpos[3:7]}")
    print(f"  target: {env.target_pos}")
    print()
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        start_time = time.time()
        
        while viewer.is_running() and step < 2000:
            pos = env.data.qpos[:3].copy()
            quat = env.data.qpos[3:7].copy()
            vel = env.data.qvel[:3].copy()
            ang_vel = env.data.qvel[3:6].copy()
            
            # Debug every 100 steps
            debug = (step % 200 == 0)
            if debug:
                print(f"Step {step}:")
            
            motors = pid.compute(pos, quat, vel, ang_vel, env.target_pos, debug=debug)
            
            env.data.ctrl[:4] = motors
            mujoco.mj_step(env.model, env.data)
            
            viewer.sync()
            step += 1
            
            # Check crash
            if env.data.qpos[2] < 0.1:
                print(f"\n*** CRASHED at step {step} ***")
                print(f"Final pos: {env.data.qpos[:3]}")
                print(f"Final quat: {env.data.qpos[3:7]}")
                euler = quat_to_euler(env.data.qpos[3:7])
                print(f"Final euler: ({np.degrees(euler[0]):.1f}, {np.degrees(euler[1]):.1f}, {np.degrees(euler[2]):.1f})")
                break
            
            # Check target reached
            dist = np.linalg.norm(env.target_pos - pos)
            if dist < 0.5:
                print(f"\n*** TARGET REACHED at step {step} ***")
                break
            
            # Real-time-ish
            elapsed = time.time() - start_time
            expected = step * env.dt
            if expected > elapsed:
                time.sleep(expected - elapsed)
    
    print("\nDone.")


if __name__ == "__main__":
    debug_run()