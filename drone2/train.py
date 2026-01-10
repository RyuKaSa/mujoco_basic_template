"""
Drone Training with PID Stabilization + Auto-Heading

Supports two training modes:
1. PPO (Reinforcement Learning) - learns from scratch with rewards
2. BC (Behavioral Cloning) - learns from demonstrations

Usage:
    # Standard RL training
    python train.py --mode easy --steps 500000
    
    # Behavioral Cloning from demonstrations
    python train.py --bc --demo demos.npz
    
    # BC pre-training then PPO fine-tuning
    python train.py --bc --demo demos.npz --finetune --steps 200000
    
    # Record demos first:
    python record_demo.py --episodes 10 --output demos.npz
"""

import argparse
import os
import signal
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from drone_env import DroneEnv


class CurriculumCallback(BaseCallback):
    """Gradually increases task difficulty during training."""
    
    def __init__(self, 
                 start_easy_ratio=1.0,
                 end_easy_ratio=0.3,
                 start_min_dist=3.0,
                 start_max_dist=8.0,
                 end_min_dist=5.0,
                 end_max_dist=25.0,
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
        
        easy_ratio = self.start_easy_ratio + progress * (self.end_easy_ratio - self.start_easy_ratio)
        min_dist = self.start_min_dist + progress * (self.end_min_dist - self.start_min_dist)
        max_dist = self.start_max_dist + progress * (self.end_max_dist - self.start_max_dist)
        
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
        self.crash_reasons = {}
    
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                targets = info.get('targets_reached', 0)
                self.episode_targets.append(targets)
                
                crash_reason = info.get('crash_reason', None)
                if crash_reason:
                    self.crash_reasons[crash_reason] = self.crash_reasons.get(crash_reason, 0) + 1
                
                if len(self.episode_targets) % 100 == 0 and self.verbose > 0:
                    recent = self.episode_targets[-100:]
                    print(f"Last 100 episodes: avg targets = {np.mean(recent):.1f}, max = {max(recent)}")
                    if self.crash_reasons:
                        print(f"  Crash reasons: {dict(self.crash_reasons)}")
                        self.crash_reasons = {}
        
        return True


def make_env(easy_ratio, min_dist, max_dist, dwell_time, timeout):
    """Factory for creating drone environments."""
    def _init():
        env = DroneEnv()
        env.easy_ratio = easy_ratio
        env.min_target_dist = min_dist
        env.max_target_dist = max_dist
        env.dwell_time = dwell_time
        env.target_timeout = timeout
        env = Monitor(env)
        return env
    return _init


def load_demonstrations(demo_path):
    """Load demonstrations from npz file."""
    print(f"Loading demonstrations from {demo_path}")
    data = np.load(demo_path)
    
    observations = data['observations']
    actions = data['actions']
    
    print(f"  Loaded {len(observations)} transitions")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Action shape: {actions.shape}")
    
    # Basic stats
    print(f"  Action stats: mean={actions.mean(axis=0)}, std={actions.std(axis=0)}")
    
    return observations, actions


class BCPolicy(nn.Module):
    """Simple MLP policy for Behavioral Cloning."""
    
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, act_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.network(obs)
    
    def get_action(self, obs):
        """Get action from numpy observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = self.network(obs_tensor).squeeze(0).numpy()
        return action


def train_bc(observations, actions, epochs=100, batch_size=64, lr=3e-4, hidden_dims=[256, 256]):
    """Train a policy using Behavioral Cloning."""
    print("\n" + "=" * 60)
    print("BEHAVIORAL CLONING")
    print("=" * 60)
    
    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(
        torch.FloatTensor(observations),
        torch.FloatTensor(actions)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create policy
    policy = BCPolicy(obs_dim, act_dim, hidden_dims)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\nTraining on {len(observations)} transitions...")
    
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()
            pred_actions = policy(obs_batch)
            loss = criterion(pred_actions, act_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy.state_dict(), "bc_policy_best.pt")
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss = {avg_loss:.6f} (best: {best_loss:.6f})")
    
    # Load best policy
    policy.load_state_dict(torch.load("bc_policy_best.pt"))
    print(f"\nBC training complete. Best loss: {best_loss:.6f}")
    
    return policy


def evaluate_bc_policy(policy, env, n_episodes=10):
    """Evaluate BC policy on environment."""
    print(f"\nEvaluating BC policy on {n_episodes} episodes...")
    
    total_targets = 0
    total_steps = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 5000:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        targets = info.get('targets_reached', 0)
        total_targets += targets
        total_steps += steps
        print(f"  Episode {ep+1}: {targets} targets in {steps} steps")
    
    print(f"\nAverage: {total_targets/n_episodes:.1f} targets, {total_steps/n_episodes:.0f} steps")
    return total_targets / n_episodes


def copy_bc_to_ppo(bc_policy, ppo_model):
    """Copy BC policy weights to PPO policy network."""
    print("\nCopying BC weights to PPO policy...")
    
    # Get PPO's policy network
    ppo_policy = ppo_model.policy
    
    # The PPO policy has mlp_extractor and action_net
    # BC policy has a single network
    # We need to map BC layers to PPO's mlp_extractor.policy_net and action_net
    
    bc_state = bc_policy.state_dict()
    
    # BC network structure: [Linear, ReLU, Linear, ReLU, Linear, Tanh]
    # Indices: 0, 2, 4 are the Linear layers
    
    # PPO mlp_extractor.policy_net has the hidden layers
    # PPO action_net has the final layer
    
    try:
        # Copy hidden layers to mlp_extractor.policy_net
        ppo_policy.mlp_extractor.policy_net[0].weight.data = bc_state['network.0.weight']
        ppo_policy.mlp_extractor.policy_net[0].bias.data = bc_state['network.0.bias']
        ppo_policy.mlp_extractor.policy_net[2].weight.data = bc_state['network.2.weight']
        ppo_policy.mlp_extractor.policy_net[2].bias.data = bc_state['network.2.bias']
        
        # Copy output layer to action_net
        ppo_policy.action_net.weight.data = bc_state['network.4.weight']
        ppo_policy.action_net.bias.data = bc_state['network.4.bias']
        
        print("Successfully copied BC weights to PPO!")
    except Exception as e:
        print(f"Warning: Could not copy all weights: {e}")
        print("PPO will start from scratch.")


def train_ppo(args, env, eval_env, bc_policy=None):
    """Train with PPO (optionally initialized from BC)."""
    print("\n" + "=" * 60)
    print("PPO TRAINING" + (" (initialized from BC)" if bc_policy else ""))
    print("=" * 60)
    
    # PPO hyperparameters
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
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
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs",
        )
        
        # Copy BC weights if available
        if bc_policy is not None:
            copy_bc_to_ppo(bc_policy, model)
    
    # Callbacks
    callbacks = []
    
    callbacks.append(CheckpointCallback(
        save_freq=50000 // args.n_envs,
        save_path="./checkpoints/",
        name_prefix="drone",
        save_vecnormalize=True,
    ))
    
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=25000 // args.n_envs,
        n_eval_episodes=10,
        deterministic=True,
    ))
    
    callbacks.append(MetricsCallback(verbose=1))
    
    if args.mode == "curriculum":
        callbacks.append(CurriculumCallback(
            start_easy_ratio=1.0,
            end_easy_ratio=0.3,
            start_min_dist=3.0,
            start_max_dist=8.0,
            end_min_dist=5.0,
            end_max_dist=args.max_dist,
            curriculum_steps=args.steps // 2,
            verbose=1,
        ))
    
    # Graceful shutdown
    def save_and_exit(signum, frame):
        print(f"\n\nSaving model...")
        model.save("drone_model")
        env.save("drone_model_vecnormalize.pkl")
        print("Saved: drone_model.zip, drone_model_vecnormalize.pkl")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)
    
    # Train
    print(f"\nTraining for {args.steps:,} steps...")
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


def main(args):
    print("=" * 60)
    print("DRONE TRAINING")
    print("=" * 60)
    
    # Determine settings
    if args.mode == "easy":
        easy_ratio = 1.0
    elif args.mode == "hard":
        easy_ratio = 0.0
    else:
        easy_ratio = 1.0  # curriculum starts easy
    
    min_dist = args.min_dist
    max_dist = args.max_dist
    
    if args.mode == "curriculum":
        min_dist = 3.0
        max_dist = 8.0
    
    print(f"Mode: {args.mode}")
    print(f"BC: {args.bc}, Demo: {args.demo}")
    print(f"Distance range: [{min_dist}, {max_dist}] m")
    
    # Create environments
    if args.bc and not args.finetune:
        # BC only - single env
        env = DroneEnv()
        env.easy_ratio = easy_ratio
        env.min_target_dist = min_dist
        env.max_target_dist = max_dist
        env.dwell_time = args.dwell
        env.target_timeout = args.timeout
        eval_env = env
    else:
        # PPO needs vectorized envs
        env = SubprocVecEnv([
            make_env(easy_ratio, min_dist, max_dist, args.dwell, args.timeout) 
            for _ in range(args.n_envs)
        ])
        
        if args.resume and os.path.exists(f"{args.resume}_vecnormalize.pkl"):
            env = VecNormalize.load(f"{args.resume}_vecnormalize.pkl", env)
            env.training = True
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        
        eval_env = SubprocVecEnv([make_env(0.5, 5.0, 15.0, args.dwell, args.timeout)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    bc_policy = None
    
    # Behavioral Cloning
    if args.bc:
        if not args.demo:
            print("ERROR: --bc requires --demo <path>")
            return
        
        observations, actions = load_demonstrations(args.demo)
        
        bc_policy = train_bc(
            observations, actions,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch,
            lr=args.bc_lr,
        )
        
        # Save BC policy
        torch.save(bc_policy.state_dict(), "bc_policy.pt")
        print("Saved BC policy to bc_policy.pt")
        
        # Evaluate
        eval_env_single = DroneEnv()
        eval_env_single.easy_ratio = easy_ratio
        eval_env_single.min_target_dist = min_dist
        eval_env_single.max_target_dist = max_dist
        evaluate_bc_policy(bc_policy, eval_env_single, n_episodes=5)
        eval_env_single.close()
        
        if not args.finetune:
            print("\nBC training complete. Use --finetune to continue with PPO.")
            return
    
    # PPO training (or fine-tuning after BC)
    if not args.bc or args.finetune:
        train_ppo(args, env, eval_env, bc_policy=bc_policy)
    
    env.close()
    if eval_env is not env:
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train drone with BC and/or PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training mode
    parser.add_argument("--bc", action="store_true",
                        help="Use Behavioral Cloning")
    parser.add_argument("--demo", type=str, default=None,
                        help="Path to demonstration file (.npz)")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune with PPO after BC")
    
    # BC settings
    parser.add_argument("--bc-epochs", type=int, default=100,
                        help="BC training epochs")
    parser.add_argument("--bc-batch", type=int, default=64,
                        help="BC batch size")
    parser.add_argument("--bc-lr", type=float, default=3e-4,
                        help="BC learning rate")
    
    # PPO settings
    parser.add_argument("--steps", type=int, default=1000000,
                        help="PPO training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume PPO from checkpoint")
    
    # Environment
    parser.add_argument("--mode", type=str, default="easy",
                        choices=["easy", "hard", "curriculum"])
    parser.add_argument("--min-dist", type=float, default=5.0)
    parser.add_argument("--max-dist", type=float, default=20.0)
    parser.add_argument("--dwell", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    
    args = parser.parse_args()
    
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./best_model", exist_ok=True)
    os.makedirs("./eval_logs", exist_ok=True)
    
    main(args)