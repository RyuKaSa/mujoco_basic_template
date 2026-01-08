"""
MuJoCo Test Script for PID-based Drone Environment

Run this to test the PID layer with your actual MuJoCo model.
This will help you tune the PID gains for your specific drone.

Usage:
    python test_pid_mujoco.py

Requirements:
    - model.xml in the same directory
    - motor_mixer.py in the same directory
    - simple_pid installed (pip install simple-pid)
"""

import numpy as np
import time

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: mujoco not installed. Run: pip install mujoco")
    exit(1)

try:
    from simple_pid import PID
except ImportError:
    print("ERROR: simple_pid not installed. Run: pip install simple-pid")
    exit(1)

try:
    from motor_mixer import mix  # Keep for reference, but not used
except ImportError:
    print("Note: motor_mixer.py not found - using direct motor calculation")


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


def quat_rotate_inverse(quat, vec):
    """Rotate vector by inverse of quaternion (world to body frame)."""
    w = quat[0]
    q_xyz = -quat[1:4]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


class SimplePIDController:
    """
    PID controller matching the reference code structure.
    
    Key insight from reference code:
        cmd_thrust = self.pid_alt(alt) + 4.08
        
    So the altitude PID outputs an ADJUSTMENT to the base hover thrust (4.08).
    The mixer then does: motor_thrust = base + adjustment
    """
    
    def __init__(self, dt=0.002):
        self.dt = dt
        
        # Inner loop PIDs (from reference)
        self.pid_alt = PID(21.0, 0.0, 0.05, setpoint=0, output_limits=(-3.0, 3.0))
        self.pid_roll = PID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_pitch = PID(2.5, 0.2, 0.8, setpoint=0, output_limits=(-1.0, 1.0))
        self.pid_yaw = PID(0.75, 0.0, 0.4, setpoint=0, output_limits=(-1.0, 1.0))
        
        # Outer loop PIDs (from reference)
        self.pid_vx = PID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        self.pid_vy = PID(0.2, 0.0, 0.03, setpoint=0, output_limits=(-0.262, 0.262))
        
        self.outer_counter = 0
        self.outer_divider = 20
        
        # CRITICAL: Base hover thrust from reference code
        self.HOVER_THRUST = 4.0712
        
    def reset(self):
        for pid in [self.pid_alt, self.pid_roll, self.pid_pitch, 
                    self.pid_yaw, self.pid_vx, self.pid_vy]:
            pid.reset()
        self.outer_counter = 0
        
    def compute(self, pos, quat, vel_world, ang_vel, cmd_vx=0, cmd_vy=0, cmd_vz=0, cmd_yaw=0):
        """
        Compute motor commands directly (bypassing mixer's thrust_base).
        
        Returns: [motor_fr, motor_fl, motor_rl, motor_rr]
        """
        vel_body = quat_rotate_inverse(quat, vel_world)
        euler = quat_to_euler(quat)
        roll, pitch, yaw = euler
        
        # Set velocity setpoints from RL commands
        self.pid_vx.setpoint = cmd_vx
        self.pid_vy.setpoint = cmd_vy
        self.pid_alt.setpoint = cmd_vz  # Vertical velocity setpoint
        self.pid_yaw.setpoint = cmd_yaw
        
        # Outer loop (every N steps): velocity -> attitude setpoints
        self.outer_counter += 1
        if self.outer_counter >= self.outer_divider:
            self.outer_counter = 0
            # From reference: angle_pitch = self.pid_v_x(v[0])
            desired_pitch = self.pid_vx(vel_body[0])
            # From reference: angle_roll = -self.pid_v_y(v[1])
            desired_roll = -self.pid_vy(vel_body[1])
            
            self.pid_pitch.setpoint = desired_pitch
            self.pid_roll.setpoint = desired_roll
        
        # Inner loop: attitude stabilization
        # From reference: cmd_thrust = self.pid_alt(alt) + 4.08
        # But we control velocity, not position, so:
        alt_adjustment = self.pid_alt(vel_world[2])
        cmd_thrust = self.HOVER_THRUST + alt_adjustment
        
        # From reference (with sign adjustments for this model):
        cmd_roll = -self.pid_roll(roll)
        cmd_pitch = -self.pid_pitch(pitch)
        cmd_yaw_out = -self.pid_yaw(ang_vel[2])
        
        # Direct motor mixing (from reference code):
        # motor_fr = thrust + roll + pitch - yaw
        # motor_fl = thrust - roll + pitch + yaw
        # motor_rl = thrust - roll - pitch - yaw
        # motor_rr = thrust + roll - pitch + yaw
        motor_fr = cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw_out
        motor_fl = cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw_out
        motor_rl = cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw_out
        motor_rr = cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw_out
        
        motors = np.array([motor_fr, motor_fl, motor_rl, motor_rr])
        return np.clip(motors, 0.0, 12.0)


def test_hover():
    """Test: Can the drone hover in place?"""
    print("\n" + "=" * 60)
    print("TEST 1: HOVER")
    print("Goal: Drone should maintain altitude with zero velocity commands")
    print("=" * 60)
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    controller = SimplePIDController(dt=model.opt.timestep)
    
    # Start at 1m altitude
    data.qpos[2] = 1.0
    data.qpos[3] = 1.0  # quat w
    mujoco.mj_forward(model, data)
    controller.reset()
    
    print(f"\nStarting at altitude: {data.qpos[2]:.2f}m")
    print(f"Hover thrust: {controller.HOVER_THRUST:.4f}")
    print("Running for 5 seconds with zero velocity commands...\n")
    
    start_z = data.qpos[2]
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        step = 0

        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body("drone").id
        cam.distance = 4.0
        cam.azimuth = 90
        cam.elevation = -20
        
        while viewer.is_running() and time.time() - start_time < 5:
            # Get state
            pos = data.qpos[:3]
            quat = data.qpos[3:7]
            vel = data.qvel[:3]
            ang_vel = data.qvel[3:6]
            
            # Zero velocity commands - controller outputs motors directly
            motors = controller.compute(pos, quat, vel, ang_vel, 
                                         cmd_vx=0, cmd_vy=0, cmd_vz=0, cmd_yaw=0)
            
            data.ctrl[:4] = motors
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Print status every 0.5s
            if step % int(0.5 / model.opt.timestep) == 0:
                euler = quat_to_euler(quat)
                print(f"t={time.time()-start_time:.1f}s: alt={pos[2]:.2f}m, "
                      f"vz={vel[2]:.2f}m/s, "
                      f"tilt={np.degrees(np.sqrt(euler[0]**2 + euler[1]**2)):.1f}°, "
                      f"motors={motors.mean():.2f}")
            
            step += 1
            
            # Time keeping
            time_until_next = model.opt.timestep - (time.time() - start_time - step * model.opt.timestep)
            if time_until_next > 0:
                time.sleep(time_until_next * 0.8)
    
    final_z = data.qpos[2]
    print(f"\nResult: Started at {start_z:.2f}m, ended at {final_z:.2f}m")
    print(f"Altitude change: {final_z - start_z:.2f}m")
    
    if abs(final_z - start_z) < 0.5 and final_z > 0.5:
        print("✓ HOVER TEST PASSED")
    else:
        print("✗ HOVER TEST FAILED - adjust HOVER_THRUST or pid_alt gains")


def test_climb():
    """Test: Can the drone climb with a positive vertical velocity command?"""
    print("\n" + "=" * 60)
    print("TEST 2: CLIMB")
    print("Goal: Drone should climb when given positive vz command")
    print("=" * 60)
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    controller = SimplePIDController(dt=model.opt.timestep)
    
    data.qpos[2] = 1.0
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    controller.reset()
    
    print(f"\nStarting at altitude: {data.qpos[2]:.2f}m")
    print("Commanding vz = 0.5 m/s for 5 seconds...\n")
    
    start_z = data.qpos[2]
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        step = 0

        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body("drone").id
        cam.distance = 4.0
        cam.azimuth = 90
        cam.elevation = -20
        
        while viewer.is_running() and time.time() - start_time < 5:
            pos = data.qpos[:3]
            quat = data.qpos[3:7]
            vel = data.qvel[:3]
            ang_vel = data.qvel[3:6]
            
            # Climb command - controller outputs motors directly
            motors = controller.compute(pos, quat, vel, ang_vel,
                                         cmd_vx=0, cmd_vy=0, cmd_vz=0.5, cmd_yaw=0)
            
            data.ctrl[:4] = motors
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            if step % int(0.5 / model.opt.timestep) == 0:
                print(f"t={time.time()-start_time:.1f}s: alt={pos[2]:.2f}m, vz={vel[2]:.2f}m/s, motors={motors.mean():.2f}")
            
            step += 1
            
            time_until_next = model.opt.timestep - (time.time() - start_time - step * model.opt.timestep)
            if time_until_next > 0:
                time.sleep(time_until_next * 0.8)
    
    final_z = data.qpos[2]
    print(f"\nResult: Started at {start_z:.2f}m, ended at {final_z:.2f}m")
    print(f"Climbed: {final_z - start_z:.2f}m")
    
    if final_z > start_z + 0.5:
        print("✓ CLIMB TEST PASSED")
    else:
        print("✗ CLIMB TEST FAILED - adjust pid_alt gains")


def test_forward():
    """Test: Can the drone move forward with a positive vx command?"""
    print("\n" + "=" * 60)
    print("TEST 3: FORWARD MOVEMENT")
    print("Goal: Drone should move forward when given positive vx command")
    print("=" * 60)
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    controller = SimplePIDController(dt=model.opt.timestep)
    
    data.qpos[2] = 2.0  # Higher start for safety
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    controller.reset()
    
    print(f"\nStarting position: x={data.qpos[0]:.2f}m, alt={data.qpos[2]:.2f}m")
    print("Commanding vx = 1.0 m/s for 7 seconds...\n")
    
    start_x = data.qpos[0]
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        step = 0

        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body("drone").id
        cam.distance = 4.0
        cam.azimuth = 90
        cam.elevation = -20
    
        while viewer.is_running() and time.time() - start_time < 7:
            pos = data.qpos[:3]
            quat = data.qpos[3:7]
            vel = data.qvel[:3]
            ang_vel = data.qvel[3:6]
            
            # Forward command - controller outputs motors directly
            motors = controller.compute(pos, quat, vel, ang_vel,
                                         cmd_vx=1.0, cmd_vy=0, cmd_vz=0, cmd_yaw=0)
            
            data.ctrl[:4] = motors
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            if step % int(0.5 / model.opt.timestep) == 0:
                vel_body = quat_rotate_inverse(quat, vel)
                euler = quat_to_euler(quat)
                print(f"t={time.time()-start_time:.1f}s: x={pos[0]:.2f}m, vx_body={vel_body[0]:.2f}m/s, "
                      f"pitch={np.degrees(euler[1]):.1f}°, alt={pos[2]:.2f}m")
            
            step += 1
            
            # Ground crash check
            if pos[2] < 0.1:
                print("\n✗ CRASHED - drone hit the ground!")
                break
            
            time_until_next = model.opt.timestep - (time.time() - start_time - step * model.opt.timestep)
            if time_until_next > 0:
                time.sleep(time_until_next * 0.8)
    
    final_x = data.qpos[0]
    print(f"\nResult: Started at x={start_x:.2f}m, ended at x={final_x:.2f}m")
    print(f"Moved forward: {final_x - start_x:.2f}m")
    
    if final_x > start_x + 1.0 and data.qpos[2] > 0.5:
        print("✓ FORWARD TEST PASSED")
    else:
        print("✗ FORWARD TEST FAILED - adjust pid_vx or pid_pitch gains")


if __name__ == "__main__":
    print("=" * 60)
    print("PID CONTROLLER TEST SUITE FOR MUJOCO")
    print("=" * 60)
    print("\nThis will run 3 tests to verify your PID gains work with MuJoCo.")
    print("Press ESC or close the viewer window to skip to the next test.\n")
    
    try:
        test_hover()
    except Exception as e:
        print(f"Hover test error: {e}")
    
    try:
        test_climb()
    except Exception as e:
        print(f"Climb test error: {e}")
    
    try:
        test_forward()
    except Exception as e:
        print(f"Forward test error: {e}")
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
    print("""
If tests failed, try adjusting gains:

HOVER FAILED:
  - Drone falls: Increase pid_alt.Kp or check hover thrust in mixer
  - Drone oscillates vertically: Increase pid_alt.Kd, decrease pid_alt.Kp
  - Drone drifts sideways: Check pid_roll and pid_pitch

CLIMB FAILED:
  - Doesn't climb: Increase pid_alt output_limits or Kp
  - Climbs too fast then oscillates: Increase Kd

FORWARD FAILED:
  - Doesn't pitch: Increase pid_vx.Kp or output_limits
  - Crashes: Decrease pid_vx output_limits, check pid_pitch
  - Oscillates: Increase pid_pitch.Kd
""")