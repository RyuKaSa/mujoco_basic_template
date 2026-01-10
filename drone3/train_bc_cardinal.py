"""
Behavioral Cloning for Cardinal Directions

Train BC on 6-direction demos, evaluate on CardinalDroneEnv.

Usage:
    python train_bc_cardinal.py                              # Train
    python train_bc_cardinal.py --eval-only --visualize      # Evaluate
"""

import argparse
import signal
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import mujoco
import mujoco.viewer

from drone_env_simple import CardinalDroneEnv, quat_to_euler


# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


class BCPolicy(nn.Module):
    """MLP: observation -> motor commands"""
    
    def __init__(self, obs_dim=13, act_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_action(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            return self.net(obs_t).squeeze(0).numpy()


def save_checkpoint(policy, optimizer, epoch, best_val_loss, filename="bc_checkpoint.pt"):
    """Save full training state for resuming."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, filename)
    print(f"  [Checkpoint saved: {filename} @ epoch {epoch+1}]")


def load_checkpoint(policy, optimizer, filename="bc_checkpoint.pt"):
    """Load training state. Returns (start_epoch, best_val_loss)."""
    try:
        checkpoint = torch.load(filename)
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from {filename} at epoch {start_epoch}")
        return start_epoch, best_val_loss
    except FileNotFoundError:
        print(f"No checkpoint found at {filename}, starting fresh")
        return 0, float('inf')
    except KeyError:
        # Old format (just state dict)
        print(f"Loading old-format weights from {filename}")
        policy.load_state_dict(torch.load(filename))
        return 0, float('inf')


def train(args):
    global SHUTDOWN_REQUESTED
    
    print("=" * 60)
    print("BC TRAINING - CARDINAL DIRECTIONS")
    print("=" * 60)
    
    # Load demos
    data = np.load(args.demos)
    obs = data['observations']
    actions = data['actions']
    
    print(f"Loaded {len(obs)} transitions from {args.demos}")
    print(f"  Obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  Act shape: {actions.shape}, range: [{actions.min():.2f}, {actions.max():.2f}]")
    print(f"  Act mean per dim: {actions.mean(axis=0)}")
    
    # Create dataset
    dataset = TensorDataset(
        torch.FloatTensor(obs),
        torch.FloatTensor(actions)
    )
    
    # Split: 90% train, 10% val
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {train_size}, Val: {val_size} (ratio: {args.train_ratio})")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Model
    policy = BCPolicy(obs_dim=obs.shape[1], act_dim=actions.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(policy, optimizer, "bc_checkpoint.pt")
    
    criterion = nn.MSELoss()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        global SHUTDOWN_REQUESTED
        if SHUTDOWN_REQUESTED:
            print("\n\nForce quit!")
            sys.exit(1)
        print("\n\n[Ctrl+C] Finishing current epoch and saving...")
        SHUTDOWN_REQUESTED = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    total_epochs = start_epoch + args.epochs
    print(f"\nTraining from epoch {start_epoch+1} to {total_epochs}...")
    print(f"  Checkpoint every {args.save_every} epochs")
    print(f"  Press Ctrl+C to save and exit gracefully\n")
    
    for epoch in range(start_epoch, total_epochs):
        # Check for shutdown request
        if SHUTDOWN_REQUESTED:
            print(f"\nShutdown requested at epoch {epoch+1}")
            save_checkpoint(policy, optimizer, epoch - 1, best_val_loss)
            torch.save(policy.state_dict(), "bc_cardinal.pt")
            print("Saved bc_cardinal.pt (current best weights)")
            return policy
        
        # Train
        policy.train()
        train_loss = 0
        for obs_batch, act_batch in train_loader:
            pred = policy(obs_batch)
            loss = criterion(pred, act_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        policy.eval()
        val_loss = 0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                pred = policy(obs_batch)
                loss = criterion(pred, act_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(policy.state_dict(), "bc_cardinal_best.pt")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(policy, optimizer, epoch, best_val_loss)
        
        # Logging
        if (epoch + 1) % 20 == 0 or epoch == start_epoch:
            print(f"Epoch {epoch+1:3d}/{total_epochs}: train={train_loss:.6f}, val={val_loss:.6f} (best={best_val_loss:.6f})")
    
    # Final save
    policy.load_state_dict(torch.load("bc_cardinal_best.pt"))
    torch.save(policy.state_dict(), "bc_cardinal.pt")
    save_checkpoint(policy, optimizer, total_epochs - 1, best_val_loss)
    
    print(f"\nDone! Best val loss: {best_val_loss:.6f}")
    print("Saved: bc_cardinal.pt, bc_cardinal_best.pt, bc_checkpoint.pt")
    
    return policy


def evaluate(args):
    print("\n" + "=" * 60)
    print(f"EVALUATING BC POLICY - MODE: {args.mode.upper()}")
    print("=" * 60)
    
    # Load policy
    policy = BCPolicy()
    policy.load_state_dict(torch.load("bc_cardinal.pt"))
    policy.eval()
    
    env = CardinalDroneEnv(mode=args.mode)
    env.target_distance = args.distance
    
    # Get physics timestep
    dt = env.model.opt.timestep
    print(f"Physics dt: {dt:.4f}s ({1/dt:.1f} Hz)")
    
    # Real-time sync should match demo recording conditions
    # Demos used real-time sync ONLY when --visualize was used
    realtime = args.realtime or (args.visualize and args.match_demo_timing)
    if realtime:
        print("Real-time sync: ENABLED")
    else:
        print("Real-time sync: DISABLED (max speed)")
    
    viewer = None
    if args.visualize:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    
    # Track per-direction success
    if args.mode == "forward":
        direction_results = {'forward': {'success': 0, 'total': 0}}
    else:
        direction_results = {d: {'success': 0, 'total': 0} for d in 
                             ['forward', 'backward', 'left', 'right', 'up', 'down']}
    
    total_targets = 0
    total_crashes = 0
    
    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        
        # Match demo initial altitude based on direction
        if args.match_demo_timing:
            direction = env.current_direction
            if direction == 'down':
                env.data.qpos[2] = 7.0
            elif direction == 'up':
                env.data.qpos[2] = 1.0
            else:
                env.data.qpos[2] = 3.0
            env.data.qpos[3] = 1.0  # quat w
            mujoco.mj_forward(env.model, env.data)
            obs = env._get_obs()
        
        direction = env.current_direction
        direction_results[direction]['total'] += 1
        
        done = False
        steps = 0
        ep_targets = 0
        
        while not done and steps < 2000:
            action = policy.get_action(obs)
            
            # Denormalize: [-1,1] -> [0,12]
            motors = (action + 1.0) * 6.0
            motors = np.clip(motors, 0, 12)
            
            env.data.ctrl[:4] = motors
            mujoco.mj_step(env.model, env.data)
            env.steps += 1
            
            obs = env._get_obs()
            pos = env.data.qpos[:3]
            dist = np.linalg.norm(env.target_pos - pos)
            
            if dist < env.target_radius:
                ep_targets += 1
                direction_results[direction]['success'] += 1
                env._spawn_target()
                direction = env.current_direction
                direction_results[direction]['total'] += 1
            
            if pos[2] < 0.1:
                done = True
                total_crashes += 1
            
            steps += 1
            
            # Real-time sync (matches demo recording when visualizing)
            if viewer and viewer.is_running():
                viewer.sync()
                if realtime:
                    time.sleep(dt)
        
        total_targets += ep_targets
        status = "CRASH" if pos[2] < 0.1 else "OK"
        print(f"  Ep {ep+1}: {ep_targets} targets, {steps} steps [{status}]")
    
    if viewer:
        viewer.close()
    
    print(f"\n{'=' * 60}")
    print("RESULTS BY DIRECTION:")
    print("-" * 40)
    for d, res in direction_results.items():
        if res['total'] > 0:
            rate = res['success'] / res['total'] * 100
            print(f"  {d:10s}: {res['success']:3d}/{res['total']:3d} ({rate:5.1f}%)")
    
    print("-" * 40)
    print(f"Total targets: {total_targets}")
    print(f"Avg per episode: {total_targets/args.eval_episodes:.1f}")
    print(f"Crashes: {total_crashes}/{args.eval_episodes}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Training
    parser.add_argument("--demos", type=str, default="cardinal_demos.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--resume", action="store_true", help="Resume from bc_checkpoint.pt")
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint every N epochs")
    
    # Evaluation
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--distance", type=float, default=5.0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Force real-time sync")
    parser.add_argument("--match-demo-timing", action="store_true", 
                        help="Match demo recording conditions (initial altitude, real-time when visualizing)")
    parser.add_argument("--mode", type=str, default="cardinal", choices=["forward", "cardinal"],
                        help="forward = only forward targets, cardinal = all 6 directions")
    
    args = parser.parse_args()
    
    if not args.eval_only:
        train(args)
    
    if args.eval or args.eval_only:
        evaluate(args)