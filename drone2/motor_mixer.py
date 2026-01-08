"""
Motor Mixer - Maps high-level commands to raw motor thrusts.

Input:  [thrust, pitch, roll, yaw]  (from RL policy, each in [-1, 1])
Output: [motor_fr, motor_fl, motor_rl, motor_rr]

Motor layout (top view, + is CW rotation, - is CCW):
       Front
    FL(-) --- FR(+)
       \     /
        \   /
         X
        /   \
       /     \
    RL(+) --- RR(-)
       Back

Standard X-config mixing:
  Thrust: All motors increase equally
  Pitch:  Front motors decrease, rear increase (nose down = positive pitch command)
  Roll:   Left motors decrease, right increase (right wing down = positive roll command)  
  Yaw:    CW motors increase, CCW decrease (CW rotation = positive yaw command)

  FR = thrust - pitch + roll - yaw  (CW,  front-right)
  FL = thrust - pitch - roll + yaw  (CCW, front-left)
  RL = thrust + pitch - roll - yaw  (CW,  rear-left)
  RR = thrust + pitch + roll + yaw  (CCW, rear-right)
"""

import numpy as np


def mix(commands: np.ndarray, 
        thrust_base: float = 4.0712, 
        thrust_scale: float = 6.0,
        attitude_scale: float = 2.0) -> np.ndarray:
    """
    Convert high-level commands to motor thrusts.
    
    Args:
        commands: [thrust, pitch, roll, yaw] each in [-1, 1]
        thrust_base: baseline thrust for hover (tune to your drone's weight/motors)
        thrust_scale: thrust command range (thrust_base ± thrust_scale)
        attitude_scale: how aggressively pitch/roll/yaw affect motors
    
    Returns:
        [motor_fr, motor_fl, motor_rl, motor_rr] clipped to [0, 12]
    
    Notes:
        - thrust=0 gives hover thrust (thrust_base)
        - thrust=1 gives max thrust (thrust_base + thrust_scale)
        - thrust=-1 gives min thrust (thrust_base - thrust_scale)
        - Attitude commands add differential thrust for rotation
    """
    thrust = commands[0]
    pitch = commands[1]
    roll = commands[2]
    yaw = commands[3]
    
    # Convert thrust from [-1,1] to actual thrust value
    t = thrust_base + thrust * thrust_scale
    
    # Attitude adjustments (differential thrust)
    p = pitch * attitude_scale
    r = roll * attitude_scale
    y = yaw * attitude_scale
    
    # Mix to motors (X-config)
    motor_fr = t - p + r - y
    motor_fl = t - p - r + y
    motor_rl = t + p - r - y
    motor_rr = t + p + r + y
    
    motors = np.array([motor_fr, motor_fl, motor_rl, motor_rr], dtype=np.float32)
    return np.clip(motors, 0.0, 12.0)


def validate_mixer():
    """Validate that mixer behaves as expected."""
    print("=== Motor Mixer Validation ===\n")
    
    # Test cases: (name, command, expected_behavior)
    tests = [
        ("Hover (neutral)", [0, 0, 0, 0], "All motors equal at hover thrust"),
        ("Thrust up", [0.5, 0, 0, 0], "All motors increase equally"),
        ("Thrust down", [-0.5, 0, 0, 0], "All motors decrease equally"),
        ("Pitch forward", [0, 0.5, 0, 0], "Front decreases, rear increases"),
        ("Pitch backward", [0, -0.5, 0, 0], "Front increases, rear decreases"),
        ("Roll right", [0, 0, 0.5, 0], "Left decreases, right increases"),
        ("Roll left", [0, 0, -0.5, 0], "Left increases, right decreases"),
        ("Yaw CW", [0, 0, 0, 0.5], "CW motors (FR,RL) decrease, CCW (FL,RR) increase"),
        ("Yaw CCW", [0, 0, 0, -0.5], "CW motors increase, CCW decrease"),
    ]
    
    for name, cmd, expected in tests:
        motors = mix(np.array(cmd))
        print(f"{name}:")
        print(f"  Command: {cmd}")
        print(f"  Motors [FR, FL, RL, RR]: {motors}")
        print(f"  Expected: {expected}")
        print()
    
    # Verify symmetry
    print("=== Symmetry Checks ===")
    hover = mix(np.array([0, 0, 0, 0]))
    print(f"Hover motors: {hover}")
    assert np.allclose(hover, hover[0]), "Hover should be symmetric"
    print("✓ Hover is symmetric")
    
    pitch_fwd = mix(np.array([0, 0.5, 0, 0]))
    pitch_back = mix(np.array([0, -0.5, 0, 0]))
    assert np.allclose(pitch_fwd[[0,1]], pitch_back[[2,3]]), "Pitch should be symmetric"
    print("✓ Pitch is symmetric")
    
    roll_right = mix(np.array([0, 0, 0.5, 0]))
    roll_left = mix(np.array([0, 0, -0.5, 0]))
    assert np.allclose(roll_right[[0,3]], roll_left[[1,2]]), "Roll should be symmetric"
    print("✓ Roll is symmetric")
    
    print("\n=== All Validations Passed ===")


if __name__ == "__main__":
    validate_mixer()