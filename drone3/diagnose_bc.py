"""
Diagnose BC training issues.
Checks demo data distribution and policy outputs.
"""

import numpy as np
import torch
from train_bc_cardinal import BCPolicy


def check_demos(demo_file="cardinal_demos.npz"):
    print("=" * 60)
    print(f"DEMO DATA ANALYSIS: {demo_file}")
    print("=" * 60)
    
    data = np.load(demo_file)
    obs = data['observations']
    actions = data['actions']
    
    print(f"\nDataset size: {len(obs)} transitions")
    
    print(f"\n--- OBSERVATIONS (shape: {obs.shape}) ---")
    print(f"  Min per dim:  {obs.min(axis=0)}")
    print(f"  Max per dim:  {obs.max(axis=0)}")
    print(f"  Mean per dim: {obs.mean(axis=0)}")
    print(f"  Std per dim:  {obs.std(axis=0)}")
    
    print(f"\n--- ACTIONS (shape: {actions.shape}) ---")
    print(f"  Min per dim:  {actions.min(axis=0)}")
    print(f"  Max per dim:  {actions.max(axis=0)}")
    print(f"  Mean per dim: {actions.mean(axis=0)}")
    print(f"  Std per dim:  {actions.std(axis=0)}")
    
    # Check for extreme values
    extreme_low = (actions < -0.9).sum(axis=0)
    extreme_high = (actions > 0.9).sum(axis=0)
    print(f"\n  Actions < -0.9 (per motor): {extreme_low}")
    print(f"  Actions > 0.9 (per motor):  {extreme_high}")
    print(f"  Total extreme: {extreme_low.sum() + extreme_high.sum()} / {actions.size} ({100*(extreme_low.sum() + extreme_high.sum())/actions.size:.1f}%)")
    
    # Distribution histogram
    print(f"\n--- ACTION DISTRIBUTION ---")
    bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(4):
        hist, _ = np.histogram(actions[:, i], bins=bins)
        hist_pct = 100 * hist / len(actions)
        print(f"  Motor {i}: {hist_pct.astype(int)}")
    
    # Convert to motor values and show
    motors = (actions + 1.0) * 6.0
    print(f"\n--- MOTOR VALUES (after denorm) ---")
    print(f"  Min per motor:  {motors.min(axis=0)}")
    print(f"  Max per motor:  {motors.max(axis=0)}")
    print(f"  Mean per motor: {motors.mean(axis=0)}")
    print(f"  Std per motor:  {motors.std(axis=0)}")
    
    # Expected hover thrust is ~4.07, which is (4.07/6 - 1) = -0.32 normalized
    expected_hover_normalized = (4.07 / 6.0) - 1.0
    print(f"\n  Expected hover thrust: ~4.07 -> normalized: {expected_hover_normalized:.3f}")
    print(f"  Actual mean normalized: {actions.mean():.3f}")
    
    return obs, actions


def check_policy(policy_file="bc_cardinal.pt", demo_file="cardinal_demos.npz"):
    print("\n" + "=" * 60)
    print(f"POLICY OUTPUT ANALYSIS: {policy_file}")
    print("=" * 60)
    
    # Load policy
    policy = BCPolicy()
    policy.load_state_dict(torch.load(policy_file))
    policy.eval()
    
    # Load some demo observations
    data = np.load(demo_file)
    obs = data['observations']
    actions_true = data['actions']
    
    # Get policy predictions for all demo observations
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs)
        actions_pred = policy(obs_tensor).numpy()
    
    print(f"\n--- POLICY OUTPUTS ON DEMO OBS ---")
    print(f"  Min per dim:  {actions_pred.min(axis=0)}")
    print(f"  Max per dim:  {actions_pred.max(axis=0)}")
    print(f"  Mean per dim: {actions_pred.mean(axis=0)}")
    print(f"  Std per dim:  {actions_pred.std(axis=0)}")
    
    # Check for saturation (tanh at limits)
    saturated_low = (actions_pred < -0.99).sum()
    saturated_high = (actions_pred > 0.99).sum()
    total = actions_pred.size
    print(f"\n  Saturated at -1: {saturated_low} ({100*saturated_low/total:.1f}%)")
    print(f"  Saturated at +1: {saturated_high} ({100*saturated_high/total:.1f}%)")
    
    # Distribution
    print(f"\n--- POLICY OUTPUT DISTRIBUTION ---")
    bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(4):
        hist, _ = np.histogram(actions_pred[:, i], bins=bins)
        hist_pct = 100 * hist / len(actions_pred)
        print(f"  Motor {i}: {hist_pct.astype(int)}")
    
    # Compare to ground truth
    print(f"\n--- PREDICTION ERROR ---")
    error = actions_pred - actions_true
    print(f"  Mean error per dim: {error.mean(axis=0)}")
    print(f"  Std error per dim:  {error.std(axis=0)}")
    print(f"  MSE per dim:        {(error**2).mean(axis=0)}")
    print(f"  Total MSE:          {(error**2).mean():.6f}")
    
    # Sample some predictions
    print(f"\n--- SAMPLE PREDICTIONS (first 5) ---")
    for i in range(5):
        print(f"  True:  {actions_true[i]} -> Motors: {(actions_true[i]+1)*6}")
        print(f"  Pred:  {actions_pred[i]} -> Motors: {(actions_pred[i]+1)*6}")
        print()


def check_network_weights(policy_file="bc_cardinal.pt"):
    print("\n" + "=" * 60)
    print(f"NETWORK WEIGHT ANALYSIS")
    print("=" * 60)
    
    policy = BCPolicy()
    policy.load_state_dict(torch.load(policy_file))
    
    for name, param in policy.named_parameters():
        data = param.data.numpy()
        print(f"\n{name}: shape={data.shape}")
        print(f"  min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}")
        
        # Check for extreme weights
        extreme = (np.abs(data) > 10).sum()
        if extreme > 0:
            print(f"  WARNING: {extreme} weights with |w| > 10!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=str, default="cardinal_demos.npz")
    parser.add_argument("--policy", type=str, default="bc_cardinal.pt")
    args = parser.parse_args()
    
    check_demos(args.demos)
    check_policy(args.policy, args.demos)
    check_network_weights(args.policy)
