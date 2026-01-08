"""
Interactive PID Tuning Script for Your Drone

This script helps you tune PID values systematically, step by step.
Each step builds on the previous one.

TUNING ORDER:
1. Find hover thrust (most critical!)
2. Tune altitude PID (vertical stability)
3. Tune roll/pitch PIDs (attitude stability)
4. Tune yaw PID (rotation control)
5. Tune velocity PIDs (movement)

Run: python tune_pid_interactive.py
"""

import numpy as np
import time
import mujoco
import mujoco.viewer
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


def quat_rotate_inverse(quat, vec):
    w = quat[0]
    q_xyz = -quat[1:4]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


# ============================================================================
# STEP 1: FIND HOVER THRUST
# ============================================================================
def find_hover_thrust():
    """
    Find the thrust value that makes your drone hover.
    This is the MOST IMPORTANT value to get right.
    """
    print("\n" + "=" * 70)
    print("STEP 1: FIND HOVER THRUST")
    print("=" * 70)
    print("""
    We need to find the motor thrust that makes your drone hover in place.
    
    Instructions:
    1. Watch the drone in the viewer
    2. Press UP/DOWN arrow keys to adjust thrust (or enter values below)
    3. Goal: Find the thrust where the drone stays at constant altitude
    
    The drone should:
    - NOT climb (thrust too high)
    - NOT fall (thrust too low)
    - Stay roughly at the same height
    """)
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    
    # Start with a guess
    hover_thrust = float(input("Enter starting thrust guess (try 3.0-5.0): ") or "4.0")
    
    while True:
        # Reset drone
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 1.0  # Start at 1m
        data.qpos[3] = 1.0  # Identity quaternion
        mujoco.mj_forward(model, data)
        
        print(f"\nTesting hover thrust = {hover_thrust:.4f}")
        print("Watch the altitude... (5 seconds)")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            start_alt = data.qpos[2]

            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = model.body("drone").id
            cam.distance = 4.0
            cam.azimuth = 90
            cam.elevation = -20
            
            while viewer.is_running() and time.time() - start_time < 5:
                # Apply equal thrust to all motors
                data.ctrl[:4] = hover_thrust
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                time.sleep(model.opt.timestep * 0.5)
        
        final_alt = data.qpos[2]
        alt_change = final_alt - start_alt
        
        print(f"\nResult: Started at {start_alt:.2f}m, ended at {final_alt:.2f}m")
        print(f"Altitude change: {alt_change:+.3f}m")
        
        if alt_change > 0.1:
            print("→ Drone CLIMBED. Thrust is TOO HIGH. Try lower value.")
        elif alt_change < -0.1:
            print("→ Drone FELL. Thrust is TOO LOW. Try higher value.")
        else:
            print("→ Drone HOVERED! This thrust is good.")
        
        response = input("\nEnter new thrust value (or 'done' to save): ").strip()
        if response.lower() == 'done':
            break
        try:
            hover_thrust = float(response)
        except ValueError:
            print("Invalid input, keeping current value")
    
    print(f"\n✓ HOVER_THRUST = {hover_thrust:.4f}")
    return hover_thrust


# ============================================================================
# STEP 2: TUNE ALTITUDE PID
# ============================================================================
def tune_altitude_pid(hover_thrust):
    """
    Tune the altitude PID to control vertical velocity.
    """
    print("\n" + "=" * 70)
    print("STEP 2: TUNE ALTITUDE PID")
    print("=" * 70)
    print("""
    Now we tune the altitude PID to control vertical velocity.
    
    The PID will:
    - Input: vertical velocity error (setpoint - current vz)
    - Output: thrust adjustment (added to hover thrust)
    
    Tuning approach:
    1. Start with only Kp (Ki=0, Kd=0)
    2. Increase Kp until drone responds but oscillates
    3. Add Kd to dampen oscillations
    4. Add small Ki only if needed
    """)
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    
    # Get initial gains
    Kp = float(input("Enter Kp (start with 1.0): ") or "1.0")
    Ki = float(input("Enter Ki (start with 0.0): ") or "0.0")
    Kd = float(input("Enter Kd (start with 0.0): ") or "0.0")
    
    while True:
        pid_alt = PID(Kp, Ki, Kd, setpoint=0, output_limits=(-3.0, 3.0))
        
        # Reset
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 1.0
        data.qpos[3] = 1.0
        mujoco.mj_forward(model, data)
        pid_alt.reset()
        
        test_cmd = input("\nTest command - 'hover' (vz=0), 'climb' (vz=0.5), 'descend' (vz=-0.5): ").strip()
        if test_cmd == 'climb':
            target_vz = 0.5
        elif test_cmd == 'descend':
            target_vz = -0.5
        else:
            target_vz = 0.0
        
        pid_alt.setpoint = target_vz
        
        print(f"\nTesting: Kp={Kp}, Ki={Ki}, Kd={Kd}, target_vz={target_vz}")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()

            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = model.body("drone").id
            cam.distance = 4.0
            cam.azimuth = 90
            cam.elevation = -20
            
            while viewer.is_running() and time.time() - start_time < 8:
                pos = data.qpos[:3]
                vel = data.qvel[:3]
                
                # PID controls vertical velocity
                thrust_adj = pid_alt(vel[2])
                total_thrust = hover_thrust + thrust_adj
                
                # Apply to all motors equally
                data.ctrl[:4] = total_thrust
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                if int((time.time() - start_time) * 2) % 2 == 0:
                    t = time.time() - start_time
                    if int(t * 10) % 5 == 0:
                        print(f"t={t:.1f}s: alt={pos[2]:.2f}m, vz={vel[2]:.2f}m/s (target={target_vz}), thrust_adj={thrust_adj:.2f}")
                
                time.sleep(model.opt.timestep * 0.5)
        
        print("\nObservations:")
        print("- If oscillating: REDUCE Kp or INCREASE Kd")
        print("- If too slow: INCREASE Kp")
        print("- If steady-state error: INCREASE Ki slightly")
        
        response = input("\nEnter 'Kp Ki Kd' (e.g., '2.0 0.1 0.5') or 'done': ").strip()
        if response.lower() == 'done':
            break
        try:
            parts = response.split()
            Kp, Ki, Kd = float(parts[0]), float(parts[1]), float(parts[2])
        except:
            print("Invalid input, keeping current values")
    
    print(f"\n✓ ALTITUDE PID: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    return Kp, Ki, Kd


# ============================================================================
# STEP 3: TUNE ROLL/PITCH PID
# ============================================================================
def tune_attitude_pid(hover_thrust, alt_gains):
    """
    Tune roll and pitch PIDs for attitude stabilization.
    """
    print("\n" + "=" * 70)
    print("STEP 3: TUNE ROLL/PITCH PID (Attitude Stabilization)")
    print("=" * 70)
    print("""
    Now we tune attitude stabilization.
    
    The PID will:
    - Input: angle error (setpoint - current angle)
    - Output: differential motor thrust
    
    Test: We'll tilt the drone and see if it recovers to level.
    
    Motor mixing for attitude:
    - Roll:  Left motors vs Right motors
    - Pitch: Front motors vs Rear motors
    """)
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    
    # Altitude PID (from previous step)
    alt_Kp, alt_Ki, alt_Kd = alt_gains
    pid_alt = PID(alt_Kp, alt_Ki, alt_Kd, setpoint=0, output_limits=(-3.0, 3.0))
    
    # Get attitude gains
    Kp = float(input("Enter attitude Kp (start with 1.0): ") or "1.0")
    Ki = float(input("Enter attitude Ki (start with 0.0): ") or "0.0")
    Kd = float(input("Enter attitude Kd (start with 0.3): ") or "0.3")
    
    while True:
        pid_roll = PID(Kp, Ki, Kd, setpoint=0, output_limits=(-1.0, 1.0))
        pid_pitch = PID(Kp, Ki, Kd, setpoint=0, output_limits=(-1.0, 1.0))
        
        # Reset with a tilt
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 2.0  # Higher for safety
        
        # Add initial tilt (rotation around x-axis = roll)
        tilt_angle = 0.2  # ~11 degrees
        data.qpos[3] = np.cos(tilt_angle / 2)  # w
        data.qpos[4] = np.sin(tilt_angle / 2)  # x (roll)
        data.qpos[5] = 0  # y
        data.qpos[6] = 0  # z
        
        mujoco.mj_forward(model, data)
        pid_alt.reset()
        pid_roll.reset()
        pid_pitch.reset()
        
        print(f"\nTesting attitude: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        print("Drone starts tilted ~11 degrees. Watch if it levels out.")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()

            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = model.body("drone").id
            cam.distance = 4.0
            cam.azimuth = 90
            cam.elevation = -20
            
            while viewer.is_running() and time.time() - start_time < 8:
                pos = data.qpos[:3]
                quat = data.qpos[3:7]
                vel = data.qvel[:3]
                euler = quat_to_euler(quat)
                roll, pitch, yaw = euler
                
                # Altitude
                thrust_adj = pid_alt(vel[2])
                cmd_thrust = hover_thrust + thrust_adj
                
                # Attitude (try to level)
                cmd_roll = -pid_roll(roll)
                cmd_pitch = -pid_pitch(pitch)
                
                # Motor mixing
                motor_fr = cmd_thrust + cmd_roll + cmd_pitch
                motor_fl = cmd_thrust - cmd_roll + cmd_pitch
                motor_rl = cmd_thrust - cmd_roll - cmd_pitch
                motor_rr = cmd_thrust + cmd_roll - cmd_pitch
                
                data.ctrl[:4] = np.clip([motor_fr, motor_fl, motor_rl, motor_rr], 0, 12)
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                t = time.time() - start_time
                if int(t * 10) % 5 == 0:
                    print(f"t={t:.1f}s: roll={np.degrees(roll):+.1f}°, pitch={np.degrees(pitch):+.1f}°, alt={pos[2]:.2f}m")
                
                time.sleep(model.opt.timestep * 0.5)
                
                if pos[2] < 0.1:
                    print("CRASHED!")
                    break
        
        print("\nObservations:")
        print("- If wobbles/oscillates: REDUCE Kp or INCREASE Kd")
        print("- If doesn't level: INCREASE Kp")
        print("- If levels but drifts: Add small Ki")
        
        response = input("\nEnter 'Kp Ki Kd' or 'done': ").strip()
        if response.lower() == 'done':
            break
        try:
            parts = response.split()
            Kp, Ki, Kd = float(parts[0]), float(parts[1]), float(parts[2])
        except:
            print("Invalid input")
    
    print(f"\n✓ ATTITUDE PID: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    return Kp, Ki, Kd


# ============================================================================
# STEP 4: TUNE VELOCITY PIDs
# ============================================================================
def tune_velocity_pid(hover_thrust, alt_gains, att_gains):
    """
    Tune velocity PIDs (outer loop).
    """
    print("\n" + "=" * 70)
    print("STEP 4: TUNE VELOCITY PID (Outer Loop)")
    print("=" * 70)
    print("""
    Finally, we tune the outer loop that converts velocity commands
    to attitude setpoints.
    
    The PID will:
    - Input: velocity error (target_vx - current_vx)
    - Output: desired pitch angle (to accelerate/decelerate)
    
    Note: These gains are typically SMALL (0.05-0.3) because
    the output is an angle (radians), not thrust.
    """)
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    
    # Inner loop PIDs
    alt_Kp, alt_Ki, alt_Kd = alt_gains
    att_Kp, att_Ki, att_Kd = att_gains
    
    pid_alt = PID(alt_Kp, alt_Ki, alt_Kd, setpoint=0, output_limits=(-3.0, 3.0))
    pid_roll = PID(att_Kp, att_Ki, att_Kd, setpoint=0, output_limits=(-1.0, 1.0))
    pid_pitch = PID(att_Kp, att_Ki, att_Kd, setpoint=0, output_limits=(-1.0, 1.0))
    
    # Outer loop gains
    Kp = float(input("Enter velocity Kp (start with 0.1): ") or "0.1")
    Ki = float(input("Enter velocity Ki (start with 0.0): ") or "0.0")
    Kd = float(input("Enter velocity Kd (start with 0.02): ") or "0.02")
    max_angle = float(input("Enter max angle limit in degrees (start with 10): ") or "10")
    max_angle_rad = np.radians(max_angle)
    
    while True:
        pid_vx = PID(Kp, Ki, Kd, setpoint=0, output_limits=(-max_angle_rad, max_angle_rad))
        pid_vy = PID(Kp, Ki, Kd, setpoint=0, output_limits=(-max_angle_rad, max_angle_rad))
        
        # Reset
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 2.0
        data.qpos[3] = 1.0
        mujoco.mj_forward(model, data)
        
        for pid in [pid_alt, pid_roll, pid_pitch, pid_vx, pid_vy]:
            pid.reset()
        
        target_vx = float(input("\nTarget forward velocity (m/s, e.g., 1.0): ") or "1.0")
        pid_vx.setpoint = target_vx
        pid_vy.setpoint = 0
        
        print(f"\nTesting: Kp={Kp}, Ki={Ki}, Kd={Kd}, max_angle={max_angle}°")
        print(f"Target: vx={target_vx} m/s")
        
        outer_counter = 0
        outer_divider = 20
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            cam = viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = model.body("drone").id
            cam.distance = 4.0
            cam.azimuth = 90
            cam.elevation = -20


            while viewer.is_running() and time.time() - start_time < 10:
                pos = data.qpos[:3]
                quat = data.qpos[3:7]
                vel = data.qvel[:3]
                euler = quat_to_euler(quat)
                roll, pitch, yaw = euler
                vel_body = quat_rotate_inverse(quat, vel)
                
                # Outer loop (every N steps)
                outer_counter += 1
                if outer_counter >= outer_divider:
                    outer_counter = 0
                    desired_pitch = pid_vx(vel_body[0])
                    desired_roll = -pid_vy(vel_body[1])
                    pid_pitch.setpoint = desired_pitch
                    pid_roll.setpoint = desired_roll
                
                # Inner loop
                thrust_adj = pid_alt(vel[2])
                cmd_thrust = hover_thrust + thrust_adj
                cmd_roll = -pid_roll(roll)
                cmd_pitch = -pid_pitch(pitch)
                
                # Mix
                motor_fr = cmd_thrust + cmd_roll + cmd_pitch
                motor_fl = cmd_thrust - cmd_roll + cmd_pitch
                motor_rl = cmd_thrust - cmd_roll - cmd_pitch
                motor_rr = cmd_thrust + cmd_roll - cmd_pitch
                
                data.ctrl[:4] = np.clip([motor_fr, motor_fl, motor_rl, motor_rr], 0, 12)
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                t = time.time() - start_time
                if int(t * 10) % 5 == 0:
                    print(f"t={t:.1f}s: x={pos[0]:.1f}m, vx={vel_body[0]:.2f}m/s (target={target_vx}), pitch={np.degrees(pitch):.1f}°")
                
                time.sleep(model.opt.timestep * 0.5)
                
                if pos[2] < 0.1:
                    print("CRASHED!")
                    break
        
        print("\nObservations:")
        print("- If overshoots velocity: REDUCE Kp or INCREASE Kd")
        print("- If too slow to reach velocity: INCREASE Kp")
        print("- If oscillates: INCREASE Kd")
        print("- If crashes: REDUCE max_angle")
        
        response = input("\nEnter 'Kp Ki Kd max_angle' or 'done': ").strip()
        if response.lower() == 'done':
            break
        try:
            parts = response.split()
            Kp, Ki, Kd = float(parts[0]), float(parts[1]), float(parts[2])
            if len(parts) > 3:
                max_angle = float(parts[3])
                max_angle_rad = np.radians(max_angle)
        except:
            print("Invalid input")
    
    print(f"\n✓ VELOCITY PID: Kp={Kp}, Ki={Ki}, Kd={Kd}, max_angle={max_angle}°")
    return Kp, Ki, Kd, max_angle


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("INTERACTIVE PID TUNING FOR YOUR DRONE")
    print("=" * 70)
    print("""
    This script will guide you through tuning your PID controllers
    step by step. Each step builds on the previous one.
    
    You'll need:
    - model.xml in the current directory
    - Patience (this takes time!)
    
    Press Enter to start...
    """)
    input()
    
    # Step 1: Find hover thrust
    hover_thrust = find_hover_thrust()
    
    # Step 2: Tune altitude PID
    alt_gains = tune_altitude_pid(hover_thrust)
    
    # Step 3: Tune attitude PID
    att_gains = tune_attitude_pid(hover_thrust, alt_gains)
    
    # Step 4: Tune velocity PID
    vel_gains = tune_velocity_pid(hover_thrust, alt_gains, att_gains)
    
    # Summary
    print("\n" + "=" * 70)
    print("TUNING COMPLETE! Here are your values:")
    print("=" * 70)
    print(f"""
# Copy these into your PID controller:

HOVER_THRUST = {hover_thrust:.4f}

# Altitude PID (controls vertical velocity)
pid_alt = PID({alt_gains[0]}, {alt_gains[1]}, {alt_gains[2]}, setpoint=0, output_limits=(-3.0, 3.0))

# Attitude PIDs (stabilize roll and pitch)
pid_roll = PID({att_gains[0]}, {att_gains[1]}, {att_gains[2]}, setpoint=0, output_limits=(-1.0, 1.0))
pid_pitch = PID({att_gains[0]}, {att_gains[1]}, {att_gains[2]}, setpoint=0, output_limits=(-1.0, 1.0))

# Yaw PID (use same as attitude, or tune separately)
pid_yaw = PID({att_gains[0] * 0.3}, 0.0, {att_gains[2] * 0.5}, setpoint=0, output_limits=(-1.0, 1.0))

# Velocity PIDs (outer loop - converts velocity to attitude)
max_angle = {np.radians(vel_gains[3]):.4f}  # {vel_gains[3]} degrees
pid_vx = PID({vel_gains[0]}, {vel_gains[1]}, {vel_gains[2]}, setpoint=0, output_limits=(-max_angle, max_angle))
pid_vy = PID({vel_gains[0]}, {vel_gains[1]}, {vel_gains[2]}, setpoint=0, output_limits=(-max_angle, max_angle))
""")
    
    # Save to file
    with open("tuned_pid_values.py", "w") as f:
        f.write(f"""# Tuned PID values for your drone
# Generated by tune_pid_interactive.py

HOVER_THRUST = {hover_thrust:.4f}

# Altitude PID
ALT_KP = {alt_gains[0]}
ALT_KI = {alt_gains[1]}
ALT_KD = {alt_gains[2]}

# Attitude PID (roll and pitch)
ATT_KP = {att_gains[0]}
ATT_KI = {att_gains[1]}
ATT_KD = {att_gains[2]}

# Velocity PID (outer loop)
VEL_KP = {vel_gains[0]}
VEL_KI = {vel_gains[1]}
VEL_KD = {vel_gains[2]}
VEL_MAX_ANGLE_DEG = {vel_gains[3]}
""")
    
    print("Values saved to: tuned_pid_values.py")


if __name__ == "__main__":
    main()
