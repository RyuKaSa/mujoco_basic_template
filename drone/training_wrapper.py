"""
Drone Training (Improved)

Key improvements:
- Better PPO hyperparameters for continuous control
- Curriculum learning support (start easy, increase difficulty)
- Proper normalization via VecNormalize
- Learning rate scheduling
- Callbacks for monitoring and curriculum

Usage:
    python train.py                          # New model, default settings
    python train.py --steps 2000000          # More steps
    python train.py --mode idle              # Hover training only
    python train.py --curriculum             # Enable curriculum learning
    python train.py --resume drone_model     # Resume training
"""

import argparse
import os
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
    Starts with mostly hovering, gradually adds more distant targets.
    """
    
    def __init__(self, 
                 start_idle_ratio=0.9,
                 end_idle_ratio=0.3,
                 start_range=1.0,
                 end_range=5.0,
                 curriculum_steps=500000,
                 verbose=0):
        super().__init__(verbose)
        self.start_idle_ratio = start_idle_ratio
        self.end_idle_ratio = end_idle_ratio
        self.start_range = start_range
        self.end_range = end_range
        self.curriculum_steps = curriculum_steps
    
    def _on_step(self) -> bool:
        # Calculate progress (0 to 1)
        progress = min(1.0, self.num_timesteps / self.curriculum_steps)
        
        # Linear interpolation
        idle_ratio = self.start_idle_ratio + progress * (self.end_idle_ratio - self.start_idle_ratio)
        target_range = self.start_range + progress * (self.end_range - self.start_range)
        
        # Update all environments
        # Note: This accesses the unwrapped envs through VecNormalize
        try:
            vec_env = self.training_env
            if hasattr(vec_env, 'venv'):  # VecNormalize wraps the actual vec env
                vec_env = vec_env.venv
            
            for env_idx in range(vec_env.num_envs):
                # SubprocVecEnv doesn't allow direct attribute access
                # We'd need to use env_method or set_attr
                vec_env.env_method('set_curriculum', idle_ratio, target_range, indices=[env_idx])
        except Exception as e:
            if self.verbose > 0:
                print(f"Curriculum update failed: {e}")
        
        # Log occasionally
        if self.num_timesteps % 50000 == 0 and self.verbose > 0:
            print(f"Curriculum: idle_ratio={idle_ratio:.2f}, range={target_range:.1f}")
        
        return True


def make_env(idle_ratio, target_range, curriculum_enabled=False):
    """Factory for creating drone environments."""
    def _init():
        env = DroneEnv()
        env.idle_ratio = idle_ratio
        env.target_range = target_range
        
        # Add curriculum method if needed
        if curriculum_enabled:
            def set_curriculum(new_idle_ratio, new_target_range):
                env.idle_ratio = new_idle_ratio
                env.target_range = new_target_range
            env.set_curriculum = set_curriculum
        
        env = Monitor(env)
        return env
    return _init


def train(args):
    print("=" * 50)
    print("Drone Training")
    print("=" * 50)
    
    # Determine idle_ratio and target_range based on mode
    if args.mode == "idle":
        idle_ratio = 1.0
        target_range = 1.0
    elif args.mode == "random":
        idle_ratio = 0.0
        target_range = args.range
    else:  # mix
        idle_ratio = 0.5
        target_range = args.range
    
    # For curriculum, start easier
    if args.curriculum:
        idle_ratio = 0.9  # Start with mostly hovering
        target_range = 1.0  # Start with close targets
        print("Curriculum learning enabled")
    
    print(f"Mode: {args.mode}")
    print(f"Initial idle_ratio: {idle_ratio}")
    print(f"Initial target_range: {target_range}")
    print(f"Total steps: {args.steps}")
    print("=" * 50)
    
    # Create parallel environments
    n_envs = args.n_envs
    env = SubprocVecEnv([
        make_env(idle_ratio, target_range, args.curriculum) 
        for _ in range(n_envs)
    ])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    
    # Create eval environment
    eval_env = SubprocVecEnv([make_env(0.5, 2.0, False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # PPO hyperparameters tuned for continuous control
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate networks
    )
    
    # Learning rate schedule (linear decay)
    lr_schedule = get_linear_fn(
        start=3e-4,
        end=1e-5,
        end_fraction=1.0,
    )
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
        # Also load the VecNormalize stats
        if os.path.exists(f"{args.resume}_vecnormalize.pkl"):
            env = VecNormalize.load(f"{args.resume}_vecnormalize.pkl", env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=2048,               # Steps per env before update
            batch_size=256,             # Larger batch for stability
            n_epochs=10,                # PPO epochs per update
            gamma=0.99,                 # Discount factor
            gae_lambda=0.95,            # GAE lambda
            clip_range=0.2,             # PPO clip range
            clip_range_vf=None,         # No value function clipping
            ent_coef=0.01,              # Entropy bonus (encourage exploration)
            vf_coef=0.5,                # Value function coefficient
            max_grad_norm=0.5,          # Gradient clipping
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs",
        )
    
    # Callbacks
    callbacks = []
    
    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // n_envs,
        save_path="./checkpoints/",
        name_prefix="drone",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=50000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # Curriculum (if enabled)
    if args.curriculum:
        curriculum_callback = CurriculumCallback(
            start_idle_ratio=0.9,
            end_idle_ratio=0.3,
            start_range=1.0,
            end_range=5.0,
            curriculum_steps=args.steps // 2,  # Reach full difficulty at halfway
            verbose=1,
        )
        callbacks.append(curriculum_callback)
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model and normalization stats
    model.save("drone_model")
    env.save("drone_model_vecnormalize.pkl")
    print("\nSaved: drone_model.zip and drone_model_vecnormalize.pkl")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drone with PPO")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to model to resume (without .zip)")
    parser.add_argument("--steps", type=int, default=1000000, 
                        help="Total training timesteps")
    parser.add_argument("--mode", type=str, default="mix", 
                        choices=["idle", "random", "mix"],
                        help="Training mode")
    parser.add_argument("--range", type=float, default=3.0, 
                        help="Target range for random mode")
    parser.add_argument("--n_envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./best_model", exist_ok=True)
    os.makedirs("./eval_logs", exist_ok=True)
    
    train(args)