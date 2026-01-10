"""
DAgger (Dataset Aggregation) for Drone Control

Instead of passive BC, we:
1. Run the current BC policy
2. At each state, ask the PID "what would you do here?"
3. Add (state, PID_action) to dataset
4. Retrain BC
5. Repeat

This collects data in states the BC actually visits, including recovery situations.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mujoco
import mujoco.viewer
import time
import signal
import sys

from drone_env_simple import CardinalDroneEnv, quat_to_euler, quat_rotate_inverse, quat_rotate
from pid_controller import SimplePIDController


# Global shutdown flag
SHUTDOWN_REQUESTED = False


class BCPolicy(nn.Module):
    """MLP policy for BC/DAgger."""
    
    def __init__(self, obs_dim=13, act_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        
        # Initialize final layer small
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        return torch.tanh(self.net(x))
    
    def get_action(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            return self.forward(obs_t).squeeze(0).numpy()


class ExpertPID:
    """Wrapper around PID that can be queried for any state."""
    
    def __init__(self, dt=0.005):
        self.pid = SimplePIDController(dt=dt)
        self.dt = dt
    
    def reset(self):
        self.pid.reset()
    
    def get_action(self, pos, quat, vel_world, ang_vel, target_pos):
        """Query PID for action given full state."""
        
        # Compute velocity commands toward target
        to_target = target_pos - pos
        to_target_body = quat_rotate_inverse(quat, to_target)
        distance = np.linalg.norm(to_target)
        
        # MORE AGGRESSIVE velocity commands
        # Scale by distance - push harder when far, gentle when close
        aggression = np.clip(distance / 3.0, 0.5, 2.0)
        
        cmd_vx = np.clip(to_target_body[0] * aggression, -3.0, 3.0)
        cmd_vy = np.clip(to_target_body[1] * aggression, -3.0, 3.0)
        cmd_vz = np.clip(to_target[2] * 1.0, -1.0, 1.0)
        
        # Get PID output
        motors = self.pid.compute(
            pos, quat, vel_world, ang_vel,
            cmd_vx=cmd_vx, cmd_vy=cmd_vy, cmd_vz=cmd_vz, cmd_yaw=0
        )
        
        # Normalize to [-1, 1]
        action_normalized = (motors / 6.0) - 1.0
        return action_normalized


def collect_dagger_data(env, policy, expert, num_episodes=10, max_steps=1000,
                        beta=0.5, visualize=False, add_perturbations=True):
    """
    Collect DAgger data by running policy and querying expert.
    
    Args:
        env: The drone environment
        policy: Current BC policy (can be None for pure expert)
        expert: The PID expert
        num_episodes: Number of episodes to collect
        max_steps: Max steps per episode
        beta: Probability of using expert action (1.0 = pure expert, 0.0 = pure policy)
        visualize: Show viewer
        add_perturbations: Add random velocity kicks to force corrections
    
    Returns:
        observations, actions (expert labels), stats
    """
    all_obs = []
    all_actions = []
    
    total_targets = 0
    total_crashes = 0
    
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    
    dt = env.model.opt.timestep
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        expert.reset()
        
        ep_targets = 0
        
        for step in range(max_steps):
            # === ADD PERTURBATIONS to force interesting situations ===
            if add_perturbations and step > 0 and step % 50 == 0:
                # Random velocity kick every 50 steps (~0.25s)
                kick_strength = 0.5
                env.data.qvel[0] += np.random.uniform(-kick_strength, kick_strength)
                env.data.qvel[1] += np.random.uniform(-kick_strength, kick_strength)
                env.data.qvel[2] += np.random.uniform(-kick_strength * 0.5, kick_strength * 0.5)
                # Small angular velocity perturbation
                env.data.qvel[3:6] += np.random.uniform(-0.2, 0.2, 3)
            
            # Get state info for expert
            pos = env.data.qpos[:3].copy()
            quat = env.data.qpos[3:7].copy()
            vel_world = env.data.qvel[:3].copy()
            ang_vel = env.data.qvel[3:6].copy()
            target_pos = env.target_pos.copy()
            
            # Query expert for the "correct" action
            expert_action = expert.get_action(pos, quat, vel_world, ang_vel, target_pos)
            
            # Decide which action to execute (for exploration)
            if policy is None or np.random.random() < beta:
                execute_action = expert_action
            else:
                execute_action = policy.get_action(obs)
            
            # Store (observation, expert_action) - always label with expert
            all_obs.append(obs.copy())
            all_actions.append(expert_action.copy())
            
            # Execute action
            motors = (execute_action + 1.0) * 6.0
            motors = np.clip(motors, 0, 12)
            
            env.data.ctrl[:4] = motors
            mujoco.mj_step(env.model, env.data)
            env.steps += 1
            
            # Get new observation
            obs = env._get_obs()
            pos = env.data.qpos[:3]
            
            # Check target reached
            dist = np.linalg.norm(env.target_pos - pos)
            if dist < env.target_radius:
                ep_targets += 1
                env._spawn_target()
                expert.reset()  # Reset PID integral for new target
            
            # Check crash
            if pos[2] < 0.1:
                total_crashes += 1
                break
            
            if viewer and viewer.is_running():
                viewer.sync()
                time.sleep(dt)  # Real-time
        
        total_targets += ep_targets
        status = "CRASH" if pos[2] < 0.1 else "OK"
        print(f"    Ep {ep+1}/{num_episodes}: {ep_targets} targets, {step+1} steps [{status}]")
    
    if viewer:
        viewer.close()
    
    stats = {
        'episodes': num_episodes,
        'transitions': len(all_obs),
        'targets': total_targets,
        'crashes': total_crashes,
        'targets_per_ep': total_targets / num_episodes,
    }
    
    return np.array(all_obs, dtype=np.float32), np.array(all_actions, dtype=np.float32), stats


def train_policy(policy, obs, actions, epochs=50, batch_size=64, lr=1e-4):
    """Train BC policy on aggregated dataset."""
    
    dataset = TensorDataset(torch.FloatTensor(obs), torch.FloatTensor(actions))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    
    policy.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for obs_batch, act_batch in loader:
            pred = policy(obs_batch)
            loss = criterion(pred, act_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
    
    return avg_loss


def evaluate_policy(env, policy, num_episodes=5, max_steps=1000, visualize=False):
    """Evaluate BC policy without expert."""
    
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    
    dt = env.model.opt.timestep
    total_targets = 0
    total_crashes = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_targets = 0
        
        for step in range(max_steps):
            action = policy.get_action(obs)
            
            motors = (action + 1.0) * 6.0
            motors = np.clip(motors, 0, 12)
            
            env.data.ctrl[:4] = motors
            mujoco.mj_step(env.model, env.data)
            
            obs = env._get_obs()
            pos = env.data.qpos[:3]
            
            dist = np.linalg.norm(env.target_pos - pos)
            if dist < env.target_radius:
                ep_targets += 1
                env._spawn_target()
            
            if pos[2] < 0.1:
                total_crashes += 1
                break
            
            if viewer and viewer.is_running():
                viewer.sync()
                time.sleep(dt)
        
        total_targets += ep_targets
    
    if viewer:
        viewer.close()
    
    return {
        'targets': total_targets,
        'targets_per_ep': total_targets / num_episodes,
        'crashes': total_crashes,
        'crash_rate': total_crashes / num_episodes,
    }


def dagger_train(args):
    """Main DAgger training loop."""
    global SHUTDOWN_REQUESTED
    
    print("=" * 60)
    print("DAgger TRAINING")
    print("=" * 60)
    
    # Setup signal handler
    def signal_handler(signum, frame):
        global SHUTDOWN_REQUESTED
        if SHUTDOWN_REQUESTED:
            sys.exit(1)
        print("\n[Ctrl+C] Will stop after current iteration...")
        SHUTDOWN_REQUESTED = True
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize
    env = CardinalDroneEnv(mode=args.mode)
    env.target_distance = args.distance
    
    expert = ExpertPID(dt=env.model.opt.timestep)
    policy = BCPolicy()
    
    # Load initial demos if provided
    if args.initial_demos:
        print(f"\nLoading initial demos: {args.initial_demos}")
        data = np.load(args.initial_demos)
        all_obs = list(data['observations'])
        all_actions = list(data['actions'])
        print(f"  Loaded {len(all_obs)} initial transitions")
    else:
        all_obs = []
        all_actions = []
    
    # Load existing policy and dataset if resuming
    if args.resume:
        try:
            policy.load_state_dict(torch.load("dagger_policy.pt"))
            print("Resumed policy from dagger_policy.pt")
        except FileNotFoundError:
            print("No policy checkpoint found, starting fresh")
        
        # Also load existing dataset if not already loaded via --initial-demos
        if not args.initial_demos:
            try:
                data = np.load("dagger_dataset.npz")
                all_obs = list(data['observations'])
                all_actions = list(data['actions'])
                print(f"Resumed dataset from dagger_dataset.npz ({len(all_obs)} transitions)")
            except FileNotFoundError:
                print("No dataset found, starting fresh")
    
    print(f"\nSettings:")
    print(f"  DAgger iterations: {args.iterations}")
    print(f"  Episodes per iteration: {args.episodes_per_iter}")
    print(f"  Beta schedule: {args.beta_start} -> {args.beta_end}")
    print(f"  Training epochs per iteration: {args.train_epochs}")
    print(f"  Early stop threshold: {args.target_threshold} targets/ep")
    print(f"  Perturbations: {not args.no_perturbations}")
    
    # Track progress across iterations
    history = []
    
    # DAgger loop
    for iteration in range(args.iterations):
        if SHUTDOWN_REQUESTED:
            break
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")
        
        # Compute beta (probability of using expert)
        # Start high (mostly expert), decay to low (mostly policy)
        progress = iteration / max(1, args.iterations - 1)
        beta = args.beta_start + (args.beta_end - args.beta_start) * progress
        print(f"\nBeta (expert prob): {beta:.2f}")
        
        # Collect data with current policy + expert labeling
        print(f"\nCollecting data...")
        new_obs, new_actions, stats = collect_dagger_data(
            env, 
            policy if iteration > 0 or args.resume else None,  # Use policy if resuming
            expert,
            num_episodes=args.episodes_per_iter,
            max_steps=args.max_steps,
            beta=beta,
            visualize=args.visualize_collect,
            add_perturbations=not args.no_perturbations
        )
        
        print(f"  Collected {stats['transitions']} transitions")
        print(f"  Targets: {stats['targets']} ({stats['targets_per_ep']:.1f}/ep)")
        print(f"  Crashes: {stats['crashes']}")
        
        # Aggregate data
        all_obs.extend(new_obs)
        all_actions.extend(new_actions)
        
        # Convert to arrays
        obs_array = np.array(all_obs, dtype=np.float32)
        actions_array = np.array(all_actions, dtype=np.float32)
        
        print(f"\nTotal dataset: {len(obs_array)} transitions")
        
        # Train policy on aggregated data
        print(f"\nTraining policy...")
        final_loss = train_policy(
            policy, obs_array, actions_array,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        # Evaluate policy (no expert help)
        print(f"\nEvaluating policy (no expert)...")
        eval_stats = evaluate_policy(
            env, policy,
            num_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            visualize=args.visualize_eval
        )
        
        print(f"  Targets: {eval_stats['targets']} ({eval_stats['targets_per_ep']:.1f}/ep)")
        print(f"  Crashes: {eval_stats['crashes']} ({eval_stats['crash_rate']*100:.0f}%)")
        
        # Track progress
        history.append({
            'iteration': iteration + 1,
            'dataset_size': len(obs_array),
            'targets_per_ep': eval_stats['targets_per_ep'],
            'crash_rate': eval_stats['crash_rate'],
            'loss': final_loss,
        })
        
        # Print progress summary
        print(f"\n  Progress: ", end="")
        for h in history[-5:]:  # Last 5 iterations
            print(f"[{h['targets_per_ep']:.1f}] ", end="")
        print()
        
        # Save checkpoint
        torch.save(policy.state_dict(), "dagger_policy.pt")
        np.savez("dagger_dataset.npz", observations=obs_array, actions=actions_array)
        print(f"\nSaved: dagger_policy.pt, dagger_dataset.npz")
        
        # Early stopping if policy is good enough
        if eval_stats['crash_rate'] == 0 and eval_stats['targets_per_ep'] >= args.target_threshold:
            print(f"\n*** Policy achieved {eval_stats['targets_per_ep']:.1f} targets/ep (threshold: {args.target_threshold})! ***")
            if not args.no_early_stop:
                print("Stopping early. Use --no-early-stop to continue.")
                break
    
    print(f"\n{'='*60}")
    print("DAgger TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final dataset size: {len(all_obs)} transitions")
    print(f"Policy saved to: dagger_policy.pt")
    
    # Print final summary
    if history:
        print(f"\nTraining history:")
        print(f"  Best targets/ep: {max(h['targets_per_ep'] for h in history):.1f}")
        print(f"  Final targets/ep: {history[-1]['targets_per_ep']:.1f}")
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="DAgger training for drone control")
    
    # DAgger settings
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of DAgger iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=20,
                        help="Episodes to collect per iteration")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max steps per episode")
    parser.add_argument("--beta-start", type=float, default=1.0,
                        help="Initial expert probability (1.0 = pure expert)")
    parser.add_argument("--beta-end", type=float, default=0.1,
                        help="Final expert probability")
    
    # Training settings
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Training epochs per DAgger iteration")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Environment
    parser.add_argument("--mode", type=str, default="cardinal",
                        choices=["forward", "cardinal"])
    parser.add_argument("--distance", type=float, default=10.0)
    
    # Data
    parser.add_argument("--initial-demos", type=str, default=None,
                        help="Optional initial demo file to start with")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from dagger_policy.pt and dagger_dataset.npz")
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Episodes for evaluation after each iteration")
    parser.add_argument("--target-threshold", type=float, default=5.0,
                        help="Early stop when targets/ep reaches this (default: 5)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Don't stop early even if policy is good")
    
    # Visualization
    parser.add_argument("--visualize-collect", action="store_true",
                        help="Visualize data collection")
    parser.add_argument("--visualize-eval", action="store_true",
                        help="Visualize evaluation")
    parser.add_argument("--no-perturbations", action="store_true",
                        help="Disable random perturbations during collection")
    
    # Eval only mode
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing policy")
    parser.add_argument("--eval-only-episodes", type=int, default=20)
    
    args = parser.parse_args()
    
    if args.eval_only:
        print("=" * 60)
        print("EVALUATING DAgger POLICY")
        print("=" * 60)
        
        env = CardinalDroneEnv(mode=args.mode)
        env.target_distance = args.distance
        
        policy = BCPolicy()
        policy.load_state_dict(torch.load("dagger_policy.pt"))
        policy.eval()
        
        stats = evaluate_policy(
            env, policy,
            num_episodes=args.eval_only_episodes,
            max_steps=2000,
            visualize=True
        )
        
        print(f"\nResults over {args.eval_only_episodes} episodes:")
        print(f"  Targets: {stats['targets']} ({stats['targets_per_ep']:.1f}/ep)")
        print(f"  Crashes: {stats['crashes']} ({stats['crash_rate']*100:.0f}%)")
    else:
        dagger_train(args)


if __name__ == "__main__":
    main()