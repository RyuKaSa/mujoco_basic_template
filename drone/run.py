"""
Run trained drone model.

Usage:
    python run.py                    # Idle mode, default model
    python run.py --mode target      # Random target mode
    python run.py --model my.zip     # Custom model path
    python run.py --no-normalize     # Skip VecNormalize (if trained without it)
"""

import argparse
import time
import os
import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium_wrapper import DroneEnv


class NormalizedEnvWrapper:
    """
    Wraps the environment to apply VecNormalize stats at inference time.
    This is needed because the model was trained with normalized observations.
    """
    
    def __init__(self, env, vec_normalize_path=None):
        self.env = env
        self.vec_normalize = None
        
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            # Create a dummy vec env to load normalization stats
            dummy_vec_env = DummyVecEnv([lambda: DroneEnv()])
            self.vec_normalize = VecNormalize.load(vec_normalize_path, dummy_vec_env)
            self.vec_normalize.training = False  # Don't update stats during inference
            self.vec_normalize.norm_reward = False  # Don't normalize rewards at inference
            print(f"Loaded normalization stats from {vec_normalize_path}")
        else:
            print("Warning: No VecNormalize stats found. Using raw observations.")
    
    def normalize_obs(self, obs):
        """Apply observation normalization."""
        if self.vec_normalize is not None:
            # VecNormalize expects batched observations
            obs_batch = np.array([obs])
            normalized = self.vec_normalize.normalize_obs(obs_batch)
            return normalized[0]
        return obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize_obs(obs), info
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self.normalize_obs(obs), reward, term, trunc, info
    
    def __getattr__(self, name):
        """Forward attribute access to underlying env."""
        return getattr(self.env, name)


def run(args):
    # Create environment
    env = DroneEnv()
    env.idle_ratio = 1.0 if args.mode == "idle" else 0.0
    env.target_range = args.range
    
    # Wrap with normalization if available
    if not args.no_normalize:
        # Try to find VecNormalize stats
        model_base = args.model.replace('.zip', '')
        normalize_path = f"{model_base}_vecnormalize.pkl"
        
        # Also check common locations
        if not os.path.exists(normalize_path):
            normalize_path = "drone_model_vecnormalize.pkl"
        
        env_wrapper = NormalizedEnvWrapper(env, normalize_path)
    else:
        env_wrapper = NormalizedEnvWrapper(env, None)
    
    # Load model
    model = PPO.load(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Mode: {args.mode}, Range: {args.range}")
    print(f"Observation space: {env.observation_space.shape}")
    print("-" * 40)
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            obs, _ = env_wrapper.reset()
            episode += 1
            print(f"\n[Episode {episode}] Target: {env.target_pos}")
            
            total_reward = 0
            steps = 0
            done = False
            
            while viewer.is_running() and not done:
                t0 = time.time()
                
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, term, trunc, info = env_wrapper.step(action)
                total_reward += reward
                steps += 1
                done = term or trunc
                
                # Sync viewer
                viewer.sync()
                
                # Print status occasionally
                if args.verbose and steps % 100 == 0:
                    print(f"  Step {steps}: dist={info['distance']:.2f}m, tilt={info['tilt']:.2f}")
                
                # Maintain realtime
                elapsed = time.time() - t0
                sleep_time = env.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Episode summary
            status = "CRASHED" if term else "TIMEOUT"
            print(f"  Result: {status}")
            print(f"  Steps: {steps}, Reward: {total_reward:.1f}, Final dist: {info['distance']:.2f}m")
            
            time.sleep(0.5)  # Brief pause between episodes
    
    env.close()


def test_without_model(args):
    """Test environment without a trained model (random or zero actions)."""
    env = DroneEnv()
    env.idle_ratio = 1.0 if args.mode == "idle" else 0.0
    env.target_range = args.range
    
    print("Testing environment (no model)")
    print(f"Mode: {args.mode}, Range: {args.range}")
    print("-" * 40)
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs, _ = env.reset()
            print(f"\nTarget: {env.target_pos}")
            
            total_reward = 0
            done = False
            steps = 0
            
            while viewer.is_running() and not done:
                t0 = time.time()
                
                # Zero action (hover attempt) or random
                if args.random_actions:
                    action = env.action_space.sample()
                else:
                    action = np.array([0, 0, 0, 0], dtype=np.float32)
                
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                steps += 1
                done = term or trunc
                
                viewer.sync()
                
                elapsed = time.time() - t0
                if env.dt > elapsed:
                    time.sleep(env.dt - elapsed)
            
            print(f"Steps: {steps}, Reward: {total_reward:.1f}, Final dist: {info['distance']:.2f}m")
            time.sleep(0.5)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained drone model")
    parser.add_argument("--model", type=str, default="drone_model.zip",
                        help="Path to trained model")
    parser.add_argument("--mode", type=str, default="idle", 
                        choices=["idle", "target"],
                        help="idle=hover in place, target=fly to random targets")
    parser.add_argument("--range", type=float, default=3.0,
                        help="Target range for target mode")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Don't apply VecNormalize (for models trained without it)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print status during episode")
    parser.add_argument("--test", action="store_true",
                        help="Test environment without model")
    parser.add_argument("--random-actions", action="store_true",
                        help="Use random actions instead of zero (with --test)")
    args = parser.parse_args()
    
    if args.test:
        test_without_model(args)
    else:
        if not os.path.exists(args.model):
            print(f"Error: Model not found: {args.model}")
            print("Train a model first with: python train_improved.py")
            print("Or test the environment with: python run.py --test")
            exit(1)
        run(args)