"""Debug script: 1 env, all joints zero, print all contact forces every step."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug contact forces.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Fetch-Keypoint-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import fetch_project.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, num_envs=1, device="cuda:0")
env = gym.make(args_cli.task, cfg=env_cfg)
base_env = env.unwrapped

obs, _ = env.reset()

asset = base_env.scene["robot"]
contact_sensor = base_env.scene.sensors["contact_forces"]

# Set all joints to zero position and zero velocity
num_joints = asset.data.joint_pos.shape[1]
zero_pos = torch.zeros(1, num_joints, device=base_env.device)
zero_vel = torch.zeros(1, num_joints, device=base_env.device)
asset.write_joint_state_to_sim(zero_pos, zero_vel)

# Zero action
num_actions = base_env.action_manager.action.shape[1]
zero_action = torch.zeros(1, num_actions, device=base_env.device)

print(f"\n{'='*80}")
print(f"Robot bodies ({asset.num_bodies}): {asset.body_names}")
print(f"Contact sensor bodies ({contact_sensor.num_bodies}): {contact_sensor.body_names}")
print(f"Joints ({asset.num_joints}): {asset.joint_names}")
print(f"Action dim: {num_actions}")
print(f"{'='*80}\n")

for step in range(200):
    obs, rew, terminated, truncated, info = env.step(zero_action)

    # Re-set joints to zero every step to keep robot still
    if step < 10:
        asset.write_joint_state_to_sim(zero_pos, zero_vel)

    # Get contact forces
    net_forces = contact_sensor.data.net_forces_w_history  # (N, H, B, 3)
    # Take max over history
    # Shape: (1, H, B, 3) -> for each body, get the force vector at the history step with max magnitude
    force_mag_per_hist = torch.norm(net_forces[0], dim=-1)  # (H, B)
    max_hist_idx = torch.argmax(force_mag_per_hist, dim=0)  # (B,)
    
    body_names = contact_sensor.body_names
    
    # Current step forces (latest history = index 0)
    current_forces = net_forces[0, 0, :, :]  # (B, 3) - latest timestep

    if step % 10 == 0:
        print(f"\n--- Step {step} | Reward: {rew.item():.4f} | Terminated: {terminated.item()} ---")
        print(f"  Base pos: {asset.data.root_pos_w[0].cpu().numpy()}")
        print(f"  Base quat: {asset.data.root_quat_w[0].cpu().numpy()}")
        proj_grav = asset.data.projected_gravity_b[0].cpu().numpy()
        print(f"  Projected gravity (body): {proj_grav}  tilt={torch.norm(asset.data.projected_gravity_b[0, :2]).item():.4f}")
        print(f"  Joint pos: {asset.data.joint_pos[0].cpu().numpy()}")
        print()
        print(f"  {'Body Name':<35} {'Fx':>8} {'Fy':>8} {'Fz':>8} {'|F|':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for b_idx in range(len(body_names)):
            fx, fy, fz = current_forces[b_idx].cpu().numpy()
            mag = torch.norm(current_forces[b_idx]).item()
            # Only print bodies with non-negligible force
            if mag > 0.01:
                print(f"  {body_names[b_idx]:<35} {fx:>8.2f} {fy:>8.2f} {fz:>8.2f} {mag:>8.2f}")
        
        # Also print bodies with zero force that might be interesting
        zero_bodies = [body_names[i] for i in range(len(body_names)) 
                       if torch.norm(current_forces[i]).item() <= 0.01]
        if zero_bodies:
            print(f"\n  Zero-force bodies ({len(zero_bodies)}): {', '.join(zero_bodies[:15])}")
            if len(zero_bodies) > 15:
                print(f"    ... and {len(zero_bodies)-15} more")

env.close()
simulation_app.close()
