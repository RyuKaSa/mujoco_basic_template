"""
Record PID demos for 6 cardinal directions.

Directions (body frame):
- Forward (+X)
- Backward (-X)
- Left (+Y)
- Right (-Y)
- Up (+Z)
- Down (-Z)

Each direction gets multiple runs at 5m distance.
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import time

from drone_env_simple import quat_to_euler, quat_rotate_inverse, quat_rotate
from pid_controller import SimplePIDController


# 6 cardinal directions in body frame
ALL_DIRECTIONS = {
    'forward':  np.array([ 1.0,  0.0,  0.0]),
    'backward': np.array([-1.0,  0.0,  0.0]),
    'left':     np.array([ 0.0,  1.0,  0.0]),
    'right':    np.array([ 0.0, -1.0,  0.0]),
    'up':       np.array([ 0.0,  0.0,  1.0]),
    'down':     np.array([ 0.0,  0.0, -1.0]),
}

FORWARD_ONLY = {
    'forward':  np.array([ 1.0,  0.0,  0.0]),
}


def get_obs(model, data, target_pos):
    """Get observation matching the env format."""
    pos = data.qpos[:3]
    quat = data.qpos[3:7]
    vel_world = data.qvel[:3]
    ang_vel = data.qvel[3:6]
    
    to_target = target_pos - pos
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


def record_direction(model, data, pid, direction_name, direction_body, 
                     distance, num_runs, visualize=False):
    """Record multiple runs for a single direction."""
    
    all_obs = []
    all_actions = []
    successful_runs = 0
    
    print(f"\n  Recording {direction_name.upper()}...")
    
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)
    
    for run in range(num_runs):
        # Reset drone to origin
        mujoco.mj_resetData(model, data)
        
        # Start altitude depends on direction
        if direction_name == 'down':
            data.qpos[2] = 7.0  # Higher start for down
        elif direction_name == 'up':
            data.qpos[2] = 1.0
        else:
            data.qpos[2] = 3.0  # Horizontal directions
        
        data.qpos[3] = 1.0  # quat w
        mujoco.mj_forward(model, data)
        
        # Compute target in world frame
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()
        
        if direction_name in ['up', 'down']:
            # Up/down are world frame
            target_world = direction_body * distance
            target_pos = pos + target_world
        else:
            # Horizontal directions are body frame
            target_world = quat_rotate(quat, direction_body)
            target_pos = pos + target_world * distance
        
        # Clamp altitude
        target_pos[2] = np.clip(target_pos[2], 0.5, 15.0)
        
        pid.reset()
        
        run_obs = []
        run_actions = []
        reached = False
        
        for step in range(3000):  # Max 6 seconds
            pos = data.qpos[:3]
            quat = data.qpos[3:7]
            vel = data.qvel[:3]
            ang_vel = data.qvel[3:6]
            
            # Get observation
            obs = get_obs(model, data, target_pos)
            
            # Compute velocity command toward target
            to_target = target_pos - pos
            to_target_body = quat_rotate_inverse(quat, to_target)
            
            cmd_vx = np.clip(to_target_body[0] * 0.5, -1.0, 1.0)
            cmd_vy = np.clip(to_target_body[1] * 0.5, -1.0, 1.0)
            cmd_vz = np.clip(to_target[2] * 0.5, -0.5, 0.5)
            
            # PID computes motors
            motors = pid.compute(pos, quat, vel, ang_vel,
                                 cmd_vx=cmd_vx, cmd_vy=cmd_vy, cmd_vz=cmd_vz, cmd_yaw=0)
            
            # Store normalized action
            action_normalized = (motors / 6.0) - 1.0
            run_obs.append(obs)
            run_actions.append(action_normalized)
            
            # Step physics
            data.ctrl[:4] = motors
            mujoco.mj_step(model, data)
            
            # Visualization: sync and slow down to real-time
            if viewer and viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)  # Real-time only when visualizing
            
            # Check target reached
            dist = np.linalg.norm(target_pos - data.qpos[:3])
            if dist < 0.5:
                reached = True
                break
            
            # Check crash
            if data.qpos[2] < 0.1:
                break
        
        if reached:
            all_obs.extend(run_obs)
            all_actions.extend(run_actions)
            successful_runs += 1
            print(f"    Run {run+1}/{num_runs}: SUCCESS ({step} steps)")
        else:
            print(f"    Run {run+1}/{num_runs}: FAILED")
    
    if viewer:
        viewer.close()
    
    print(f"  {direction_name}: {successful_runs}/{num_runs} successful, {len(all_obs)} transitions")
    
    return all_obs, all_actions


def main(args):
    print("=" * 60)
    print("DEMO RECORDER")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Distance: {args.distance}m")
    print(f"Runs per direction: {args.runs}")
    
    # Select directions based on mode
    if args.mode == 'forward':
        directions = FORWARD_ONLY
    else:
        directions = ALL_DIRECTIONS
    
    print(f"Directions: {list(directions.keys())}")
    
    model = mujoco.MjModel.from_xml_path("model.xml")
    data = mujoco.MjData(model)
    pid = SimplePIDController(dt=model.opt.timestep)
    
    all_obs = []
    all_actions = []
    
    for direction_name, direction_body in directions.items():
        obs, actions = record_direction(
            model, data, pid,
            direction_name, direction_body,
            args.distance, args.runs,
            visualize=args.visualize
        )
        all_obs.extend(obs)
        all_actions.extend(actions)
    
    # Save
    obs_array = np.array(all_obs, dtype=np.float32)
    act_array = np.array(all_actions, dtype=np.float32)
    
    np.savez(args.output, observations=obs_array, actions=act_array)
    
    print(f"\n{'=' * 60}")
    print(f"SAVED: {args.output}")
    print(f"  Total transitions: {len(all_obs)}")
    print(f"  Obs shape: {obs_array.shape}")
    print(f"  Action shape: {act_array.shape}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "cardinal"],
                        help="forward = only forward, cardinal = all 6 directions")
    parser.add_argument("--distance", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=10, help="Runs per direction")
    parser.add_argument("--output", type=str, default="demos.npz")
    parser.add_argument("--visualize", action="store_true")
    
    args = parser.parse_args()
    main(args)