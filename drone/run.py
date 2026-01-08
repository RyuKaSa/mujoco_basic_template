"""
Run trained drone model.

Usage:
    python run.py                        # Default model, easy targets
    python run.py --mode hard            # Hard targets (combined axes)
    python run.py --dist 1 5             # Target distance range
    python run.py --dwell 0.5            # Faster target capture
    python run.py --model best.zip       # Custom model
    python run.py --test                 # Test env without model
"""

import argparse
import time
import os
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium_wrapper import DroneEnv


def update_target_marker(model, data, target_pos):
    """Move the mocap target marker to the current target position."""
    if model.nmocap > 0:
        data.mocap_pos[0] = target_pos


def draw_bounding_box(viewer, bounds_min, bounds_max, rgba=[1, 0, 0, 0.3]):
    """Draw the bounding box as lines in the viewer."""
    # Clear previous custom geometry
    viewer.user_scn.ngeom = 0
    
    # Get corners of the box
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    
    # 12 edges of a box
    edges = [
        # Bottom face
        ([x0, y0, z0], [x1, y0, z0]),
        ([x1, y0, z0], [x1, y1, z0]),
        ([x1, y1, z0], [x0, y1, z0]),
        ([x0, y1, z0], [x0, y0, z0]),
        # Top face
        ([x0, y0, z1], [x1, y0, z1]),
        ([x1, y0, z1], [x1, y1, z1]),
        ([x1, y1, z1], [x0, y1, z1]),
        ([x0, y1, z1], [x0, y0, z1]),
        # Vertical edges
        ([x0, y0, z0], [x0, y0, z1]),
        ([x1, y0, z0], [x1, y0, z1]),
        ([x1, y1, z0], [x1, y1, z1]),
        ([x0, y1, z0], [x0, y1, z1]),
    ]
    
    for i, (p0, p1) in enumerate(edges):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
            
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=[0, 0, 0],
            pos=np.array([(p0[0]+p1[0])/2, (p0[1]+p1[1])/2, (p0[2]+p1[2])/2]),
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=np.float32)
        )
        
        # Set line endpoints using fromto
        viewer.user_scn.geoms[i].fromto = np.array([*p0, *p1], dtype=np.float64)
        viewer.user_scn.ngeom = i + 1
    
    # Also draw semi-transparent walls (optional - can be visually noisy)
    # We'll just use the lines for now


def draw_bounding_box_planes(viewer, bounds_min, bounds_max):
    """Draw bounding box as semi-transparent planes."""
    viewer.user_scn.ngeom = 0
    
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    
    cx, cy, cz = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
    sx, sy, sz = (x1-x0)/2, (y1-y0)/2, (z1-z0)/2
    
    # 6 faces: [position, size, rgba]
    # We'll draw each face as a thin box
    thickness = 0.02
    faces = [
        # Bottom (z = z0)
        ([cx, cy, z0], [sx, sy, thickness], [1, 0, 0, 0.15]),
        # Top (z = z1)
        ([cx, cy, z1], [sx, sy, thickness], [1, 0, 0, 0.15]),
        # Front (x = x1)
        ([x1, cy, cz], [thickness, sy, sz], [1, 0.3, 0, 0.15]),
        # Back (x = x0)
        ([x0, cy, cz], [thickness, sy, sz], [1, 0.3, 0, 0.15]),
        # Right (y = y1)
        ([cx, y1, cz], [sx, thickness, sz], [1, 0.6, 0, 0.15]),
        # Left (y = y0)
        ([cx, y0, cz], [sx, thickness, sz], [1, 0.6, 0, 0.15]),
    ]
    
    for i, (pos, size, rgba) in enumerate(faces):
        if i >= viewer.user_scn.maxgeom:
            break
            
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.array(size, dtype=np.float64),
            pos=np.array(pos, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=np.float32)
        )
        viewer.user_scn.ngeom = i + 1


class NormalizedEnvWrapper:
    """Applies VecNormalize stats at inference time."""
    
    def __init__(self, env, vec_normalize_path=None):
        self.env = env
        self.vec_normalize = None
        
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            dummy = DummyVecEnv([lambda: DroneEnv()])
            self.vec_normalize = VecNormalize.load(vec_normalize_path, dummy)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            print(f"Loaded normalization: {vec_normalize_path}")
    
    def normalize_obs(self, obs):
        if self.vec_normalize is not None:
            return self.vec_normalize.normalize_obs(np.array([obs]))[0]
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize_obs(obs), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self.normalize_obs(obs), reward, term, trunc, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def run(args):
    # Create environment
    env = DroneEnv()
    env.easy_ratio = 1.0 if args.mode == "easy" else 0.0 if args.mode == "hard" else 0.5
    env.min_target_dist = args.dist[0]
    env.max_target_dist = args.dist[1]
    env.dwell_time = args.dwell
    
    # Wrap with normalization
    normalize_path = args.model.replace('.zip', '') + "_vecnormalize.pkl"
    if not os.path.exists(normalize_path):
        normalize_path = "drone_model_vecnormalize.pkl"
    
    wrapper = NormalizedEnvWrapper(env, normalize_path if not args.no_norm else None)
    
    # Load model
    model = PPO.load(args.model)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode} (easy_ratio={env.easy_ratio})")
    print(f"Distance: [{env.min_target_dist}, {env.max_target_dist}] m")
    print(f"Dwell: {env.dwell_time} s")
    print(f"Walls: {'visible' if args.walls else 'hidden'}")
    print("-" * 50)
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        
        while viewer.is_running():
            obs, _ = wrapper.reset()
            episode += 1
            
            # Update target marker position
            update_target_marker(env.model, env.data, env.target_pos)
            
            # Draw bounding box walls
            if args.walls:
                if args.wall_style == "lines":
                    draw_bounding_box(viewer, env._bounds_min, env._bounds_max)
                else:
                    draw_bounding_box_planes(viewer, env._bounds_min, env._bounds_max)
            
            print(f"\n[Episode {episode}] First target: {env.target_pos}")
            if args.verbose:
                print(f"  Bounds: {env._bounds_min.round(2)} to {env._bounds_max.round(2)}")
            
            total_reward = 0
            steps = 0
            done = False
            
            while viewer.is_running() and not done:
                t0 = time.time()
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = wrapper.step(action)
                total_reward += reward
                steps += 1
                done = term or trunc
                
                viewer.sync()
                
                # Status updates
                if args.verbose and steps % 200 == 0:
                    print(f"  Step {steps}: dist={info['distance']:.2f}m, "
                          f"tilt={info['tilt']:.2f}, "
                          f"dwell={info['dwell_progress']:.0%}")
                
                if info['target_reached_this_step']:
                    # Update marker to new target position
                    update_target_marker(env.model, env.data, env.target_pos)
                    
                    # Update walls to new bounds
                    if args.walls:
                        if args.wall_style == "lines":
                            draw_bounding_box(viewer, env._bounds_min, env._bounds_max)
                        else:
                            draw_bounding_box_planes(viewer, env._bounds_min, env._bounds_max)
                    
                    print(f"  ✓ Target {info['targets_reached']}! "
                          f"New: {env.target_pos.round(2)}")
                
                # Realtime
                elapsed = time.time() - t0
                if env.dt > elapsed:
                    time.sleep(env.dt - elapsed)
            
            crash_reason = info.get('crash_reason', 'unknown')
            status = f"CRASH ({crash_reason})" if term else "TIMEOUT"
            print(f"  [{status}] Steps: {steps}, Targets: {info['targets_reached']}, "
                  f"Reward: {total_reward:.1f}")
            
            time.sleep(0.5)
    
    env.close()


def test_env(args):
    """Test environment without trained model."""
    env = DroneEnv()
    env.easy_ratio = 1.0 if args.mode == "easy" else 0.0 if args.mode == "hard" else 0.5
    env.min_target_dist = args.dist[0]
    env.max_target_dist = args.dist[1]
    env.dwell_time = args.dwell
    
    print(f"Testing environment (no model)")
    print(f"Mode: {args.mode}")
    print(f"Distance: [{env.min_target_dist}, {env.max_target_dist}] m")
    print(f"Walls: {'visible' if args.walls else 'hidden'}")
    print("-" * 50)
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        
        while viewer.is_running():
            obs, _ = env.reset()
            episode += 1
            
            # Update target marker position
            update_target_marker(env.model, env.data, env.target_pos)
            
            # Draw bounding box walls
            if args.walls:
                if args.wall_style == "lines":
                    draw_bounding_box(viewer, env._bounds_min, env._bounds_max)
                else:
                    draw_bounding_box_planes(viewer, env._bounds_min, env._bounds_max)
            
            print(f"\n[Episode {episode}] Target: {env.target_pos.round(2)}")
            if args.verbose:
                print(f"  Bounds: {env._bounds_min.round(2)} to {env._bounds_max.round(2)}")
            
            total_reward = 0
            steps = 0
            done = False
            
            while viewer.is_running() and not done:
                t0 = time.time()
                
                # Zero action (hover) or random
                if args.random:
                    action = env.action_space.sample()
                else:
                    action = np.array([0, 0, 0, 0], dtype=np.float32)
                
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                steps += 1
                done = term or trunc
                
                if info['target_reached_this_step']:
                    # Update marker to new target position
                    update_target_marker(env.model, env.data, env.target_pos)
                    
                    # Update walls to new bounds
                    if args.walls:
                        if args.wall_style == "lines":
                            draw_bounding_box(viewer, env._bounds_min, env._bounds_max)
                        else:
                            draw_bounding_box_planes(viewer, env._bounds_min, env._bounds_max)
                    
                    print(f"  ✓ Target reached! New: {env.target_pos.round(2)}")
                
                viewer.sync()
                
                elapsed = time.time() - t0
                if env.dt > elapsed:
                    time.sleep(env.dt - elapsed)
            
            crash_reason = info.get('crash_reason', 'unknown')
            status = f"CRASH ({crash_reason})" if term else "TIMEOUT"
            print(f"  [{status}] Steps: {steps}, Targets: {info['targets_reached']}, "
                  f"Reward: {total_reward:.1f}")
            time.sleep(0.5)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run trained drone model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", type=str, default="drone_model.zip",
                        help="Path to trained model")
    parser.add_argument("--mode", type=str, default="easy",
                        choices=["easy", "hard", "mix"],
                        help="Target difficulty")
    parser.add_argument("--dist", type=float, nargs=2, default=[1.0, 5.0],
                        metavar=("MIN", "MAX"),
                        help="Target distance range (meters)")
    parser.add_argument("--dwell", type=float, default=1.0,
                        help="Dwell time at target (seconds)")
    parser.add_argument("--no-norm", action="store_true",
                        help="Skip observation normalization")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print status during episode")
    
    # Wall visualization
    parser.add_argument("--walls", action="store_true",
                        help="Show bounding box walls")
    parser.add_argument("--wall-style", type=str, default="planes",
                        choices=["lines", "planes"],
                        help="Wall visualization style")
    
    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Test environment without model")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions in test mode")
    
    args = parser.parse_args()
    
    if args.test:
        test_env(args)
    else:
        if not os.path.exists(args.model):
            print(f"Error: Model not found: {args.model}")
            print("Train first: python train.py")
            print("Or test env: python run.py --test")
            exit(1)
        run(args)