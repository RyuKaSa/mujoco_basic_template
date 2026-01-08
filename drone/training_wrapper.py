"""
Drone Training

Training modes:
- easy: Cardinal direction targets only (up, down, forward, back, left, right)
- hard: Combined axis targets (forward+up, left+down, etc.)
- curriculum: Start easy, gradually increase difficulty

Usage:
    python train.py                              # Default: curriculum mode
    python train.py --mode easy --steps 500000   # Easy targets only
    python train.py --mode hard                  # Hard targets only
    python train.py --min-dist 1 --max-dist 3    # Close targets
    python train.py --dwell 0.5                  # Faster target capture
    python train.py --resume drone_model         # Resume training
"""

import argparse
import os
import signal
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from gymnasium_wrapper import DroneEnv


class CurriculumCallback(BaseCallback):
    """
    Gradually increases task difficulty during training.
    
    Progression:
    1. Start: 100% easy targets, close range (1-2m)
    2. Middle: Mix of easy/hard, medium range (1-5m)
    3. End: More hard targets, full range (1-10m)
    """
    
    def __init__(self, 
                 start_easy_ratio=1.0,
                 end_easy_ratio=0.3,
                 start_min_dist=1.0,
                 start_max_dist=2.0,
                 end_min_dist=1.0,
                 end_max_dist=10.0,
                 curriculum_steps=500000,
                 verbose=0):
        super().__init__(verbose)
        self.start_easy_ratio = start_easy_ratio
        self.end_easy_ratio = end_easy_ratio
        self.start_min_dist = start_min_dist
        self.start_max_dist = start_max_dist
        self.end_min_dist = end_min_dist
        self.end_max_dist = end_max_dist
        self.curriculum_steps = curriculum_steps
    
    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.curriculum_steps)
        
        # Linear interpolation
        easy_ratio = self.start_easy_ratio + progress * (self.end_easy_ratio - self.start_easy_ratio)
        min_dist = self.start_min_dist + progress * (self.end_min_dist - self.start_min_dist)
        max_dist = self.start_max_dist + progress * (self.end_max_dist - self.start_max_dist)
        
        # Update environments
        try:
            vec_env = self.training_env
            if hasattr(vec_env, 'venv'):
                vec_env = vec_env.venv
            
            vec_env.env_method('set_curriculum', easy_ratio, min_dist, max_dist)
        except Exception as e:
            if self.verbose > 0:
                print(f"Curriculum update failed: {e}")
        
        if self.num_timesteps % 100000 == 0 and self.verbose > 0:
            print(f"Curriculum @ {self.num_timesteps}: easy={easy_ratio:.0%}, dist=[{min_dist:.1f}, {max_dist:.1f}]")
        
        return True


class MetricsCallback(BaseCallback):
    """Log custom metrics during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_targets = []
    
    def _on_step(self) -> bool:
        # Check for episode completions
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                targets = info.get('targets_reached', 0)
                self.episode_targets.append(targets)
                
                if len(self.episode_targets) % 100 == 0 and self.verbose > 0:
                    recent = self.episode_targets[-100:]
                    print(f"Last 100 episodes: avg targets = {np.mean(recent):.1f}")
        
        return True


def make_env(easy_ratio, min_dist, max_dist, dwell_time):
    """Factory for creating drone environments."""
    def _init():
        env = DroneEnv()
        env.easy_ratio = easy_ratio
        env.min_target_dist = min_dist
        env.max_target_dist = max_dist
        env.dwell_time = dwell_time
        env = Monitor(env)
        return env
    return _init


def train(args):
    print("=" * 60)
    print("Drone Training")
    print("=" * 60)
    
    # Determine settings based on mode
    if args.mode == "easy":
        easy_ratio = 1.0
        use_curriculum = False
    elif args.mode == "hard":
        easy_ratio = 0.0
        use_curriculum = False
    else:  # curriculum (default)
        easy_ratio = 0.8
        use_curriculum = True
    
    min_dist = args.min_dist
    max_dist = args.max_dist
    dwell_time = args.dwell
    
    if use_curriculum:
        # Override to start easy
        min_dist = 1.0
        max_dist = 2.0
        print("Curriculum learning ENABLED")
    
    print(f"Mode: {args.mode}")
    print(f"Easy ratio: {easy_ratio:.0%}")
    print(f"Distance range: [{min_dist}, {max_dist}] m")
    print(f"Dwell time: {dwell_time} s")
    print(f"Total steps: {args.steps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print("=" * 60)
    
    # Create parallel environments
    env = SubprocVecEnv([
        make_env(easy_ratio, min_dist, max_dist, dwell_time) 
        for _ in range(args.n_envs)
    ])
    
    # Normalize observations and rewards
    if args.resume and os.path.exists(f"{args.resume}_vecnormalize.pkl"):
        print(f"Loading VecNormalize from {args.resume}_vecnormalize.pkl")
        env = VecNormalize.load(f"{args.resume}_vecnormalize.pkl", env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
    
    # Create eval environment (fixed difficulty for consistent eval)
    eval_env = SubprocVecEnv([make_env(0.5, 2.0, 5.0, dwell_time)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # PPO hyperparameters
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
    )
    
    lr_schedule = get_linear_fn(start=3e-4, end=1e-5, end_fraction=1.0)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=4096,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs",
        )
    
    # Callbacks
    callbacks = []
    
    # Checkpoints every 100k steps
    callbacks.append(CheckpointCallback(
        save_freq=100000 // args.n_envs,
        save_path="./checkpoints/",
        name_prefix="drone",
        save_vecnormalize=True,
    ))
    
    # Evaluation every 50k steps
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=50000 // args.n_envs,
        n_eval_episodes=10,
        deterministic=True,
    ))
    
    # Metrics logging
    callbacks.append(MetricsCallback(verbose=1))
    
    # Curriculum (if enabled)
    if use_curriculum:
        callbacks.append(CurriculumCallback(
            start_easy_ratio=1.0,
            end_easy_ratio=0.3,
            start_min_dist=1.0,
            start_max_dist=2.0,
            end_min_dist=1.0,
            end_max_dist=args.max_dist,
            curriculum_steps=args.steps // 2,
            verbose=1,
        ))
    
    # Graceful shutdown
    def save_and_exit(signum, frame):
        print(f"\n\nInterrupt received. Saving model...")
        model.save("drone_model")
        env.save("drone_model_vecnormalize.pkl")
        print("Saved: drone_model.zip, drone_model_vecnormalize.pkl")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)
    
    # Train
    print("\nStarting training...\n")
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    
    # Save final model
    model.save("drone_model")
    env.save("drone_model_vecnormalize.pkl")
    print("\nSaved: drone_model.zip, drone_model_vecnormalize.pkl")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train drone navigation with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training settings
    parser.add_argument("--steps", type=int, default=100000,
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume (without .zip)")
    
    # Mode
    parser.add_argument("--mode", type=str, default="easy",
                        choices=["easy", "hard", "curriculum"],
                        help="Training mode")
    
    # Environment settings
    parser.add_argument("--min-dist", type=float, default=20.0,
                        help="Minimum target distance (meters)")
    parser.add_argument("--max-dist", type=float, default=40.0,
                        help="Maximum target distance (meters)")
    parser.add_argument("--dwell", type=float, default=0.0,
                        help="Time to dwell at target (seconds)")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./best_model", exist_ok=True)
    os.makedirs("./eval_logs", exist_ok=True)
    
    train(args)