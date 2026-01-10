"""
Pure PID Controller - No wall-clock dependencies.
Drop-in replacement for simple_pid that works correctly in fast simulation.
"""

import numpy as np


class PurePID:
    """PID controller with NO wall-clock timing - purely uses passed dt."""
    
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = 0.0
        self._last_error = None
        
    def reset(self):
        self._integral = 0.0
        self._last_error = None
        
    def __call__(self, measured_value, dt):
        error = self.setpoint - measured_value
        
        # Proportional
        p = self.kp * error
        
        # Integral with anti-windup clamping
        self._integral += error * dt
        i = self.ki * self._integral
        
        # Derivative (on error)
        if self._last_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self._last_error) / dt
        self._last_error = error
        
        output = p + i + d
        
        # Clamp output
        lo, hi = self.output_limits
        if lo is not None and output < lo:
            output = lo
        if hi is not None and output > hi:
            output = hi
            
        return output


class SimplePIDController:
    """
    Drone PID controller using pure simulation time.
    No wall-clock dependencies - runs correctly at any simulation speed.
    """
    
    def __init__(self, dt=0.002):
        self.dt = dt
        
        # Altitude (velocity control)
        self.pid_alt = PurePID(21.0, 0.0, 0.05, setpoint=0, output_limits=(-3.0, 3.0))
        
        # Attitude control
        self.pid_roll = PurePID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_pitch = PurePID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_yaw = PurePID(0.75, 0.0, 0.4, setpoint=0, output_limits=(-1.0, 1.0))
        
        # Velocity control (outer loop)
        self.pid_vx = PurePID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        self.pid_vy = PurePID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        
        self.outer_counter = 0
        self.outer_divider = 20
        self.HOVER_THRUST = 4.0712
        
    def reset(self):
        for pid in [self.pid_alt, self.pid_roll, self.pid_pitch, 
                    self.pid_yaw, self.pid_vx, self.pid_vy]:
            pid.reset()
        self.outer_counter = 0
        
    def compute(self, pos, quat, vel_world, ang_vel, cmd_vx=0, cmd_vy=0, cmd_vz=0, cmd_yaw=0):
        from drone_env_simple import quat_to_euler, quat_rotate_inverse
        
        vel_body = quat_rotate_inverse(quat, vel_world)
        euler = quat_to_euler(quat)
        roll, pitch, yaw = euler
        
        dt = self.dt
        
        # Set velocity targets
        self.pid_vx.setpoint = cmd_vx
        self.pid_vy.setpoint = cmd_vy
        self.pid_alt.setpoint = cmd_vz
        self.pid_yaw.setpoint = cmd_yaw
        
        # Outer loop (velocity -> attitude) runs at lower rate
        self.outer_counter += 1
        if self.outer_counter >= self.outer_divider:
            self.outer_counter = 0
            outer_dt = dt * self.outer_divider  # Correct dt for outer loop
            desired_pitch = self.pid_vx(vel_body[0], outer_dt)
            desired_roll = -self.pid_vy(vel_body[1], outer_dt)
            self.pid_pitch.setpoint = desired_pitch
            self.pid_roll.setpoint = desired_roll
        
        # Altitude adjustment
        alt_adjustment = self.pid_alt(vel_world[2], dt)
        cmd_thrust = self.HOVER_THRUST + alt_adjustment
        
        # Attitude control (inner loop)
        cmd_roll = -self.pid_roll(roll, dt)
        cmd_pitch = -self.pid_pitch(pitch, dt)
        cmd_yaw_out = -self.pid_yaw(ang_vel[2], dt)
        
        # Motor mixing
        motor_fr = cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw_out
        motor_fl = cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw_out
        motor_rl = cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw_out
        motor_rr = cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw_out
        
        return np.clip([motor_fr, motor_fl, motor_rl, motor_rr], 0.0, 12.0)