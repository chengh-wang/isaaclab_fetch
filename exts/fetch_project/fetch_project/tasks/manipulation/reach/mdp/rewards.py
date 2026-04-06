# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from .utils import pose_to_keypoints, keypoint_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _check_nan(name: str, tensor: torch.Tensor, extras: dict = None):
    """Debug helper: print if any NaN/Inf found."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        print(f"[NaN DEBUG] {name}: {nan_count} NaN, {inf_count} Inf, shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}")
        if extras:
            for k, v in extras.items():
                if torch.is_tensor(v):
                    print(f"  {k}: NaN={torch.isnan(v).sum().item()}, Inf={torch.isinf(v).sum().item()}, min={v.min():.4f}, max={v.max():.4f}")

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)



def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    
    Note: This function expects the command to be in world frame (e.g., from WorldPoseCommandCfg).
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions (both in world frame)
    des_pos_w = command[:, :3]  # Command is already in world frame
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    
    Note: This function expects the command to be in world frame (e.g., from WorldPoseCommandCfg).
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions (both in world frame)
    des_pos_w = command[:, :3]  # Command is already in world frame
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    
    Note: This function expects the command to be in world frame (e.g., from WorldPoseCommandCfg).
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations (both in world frame)
    des_quat_w = command[:, 3:7]  # Command is already in world frame
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)



# =============================================================================
# Helpers
# =============================================================================

def _ee_pose(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    s = asset.data.body_state_w[:, asset_cfg.body_ids[0], :]
    _check_nan("_ee_pose", s)
    return s[:, :3], s[:, 3:7]

def _goal_pose(env, cmd_name):
    cmd = env.command_manager.get_command(cmd_name)
    _check_nan("_goal_pose", cmd)
    return cmd[:, :3], cmd[:, 3:7]


# =============================================================================
# Tracking rewards
# =============================================================================

def position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.25,
) -> torch.Tensor:
    """exp(-d_p / sigma). Range [0, 1]. 1 = perfect."""
    ee_pos, _ = _ee_pose(env, asset_cfg)
    goal_pos, _ = _goal_pose(env, command_name)
    d_p = torch.norm(ee_pos - goal_pos, dim=-1)
    result = torch.exp(-d_p / sigma)
    _check_nan("position_error_exp", result, {"ee_pos": ee_pos, "goal_pos": goal_pos, "d_p": d_p})
    return result


def position_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Raw L2 position error. Use with negative weight."""
    ee_pos, _ = _ee_pose(env, asset_cfg)
    goal_pos, _ = _goal_pose(env, command_name)
    result = torch.norm(ee_pos - goal_pos, dim=-1).clamp(max=10.0)
    _check_nan("position_error_l2", result, {"ee_pos": ee_pos, "goal_pos": goal_pos})
    return result


def position_error_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.1,
) -> torch.Tensor:
    """1 - tanh(d_p / sigma). Range [0, 1]. Fine-grained near target."""
    ee_pos, _ = _ee_pose(env, asset_cfg)
    goal_pos, _ = _goal_pose(env, command_name)
    d_p = torch.norm(ee_pos - goal_pos, dim=-1)
    result = 1.0 - torch.tanh(d_p / sigma)
    _check_nan("position_error_tanh", result, {"d_p": d_p})
    return result


def orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.25,
) -> torch.Tensor:
    """exp(-d_o / sigma). Range [0, 1]. 1 = perfect."""
    _, ee_quat = _ee_pose(env, asset_cfg)
    _, goal_quat = _goal_pose(env, command_name)
    d_o = quat_error_magnitude(ee_quat, goal_quat)
    result = torch.exp(-d_o / sigma)
    _check_nan("orientation_error_exp", result, {"ee_quat": ee_quat, "goal_quat": goal_quat, "d_o": d_o})
    return result


def orientation_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Raw orientation error in radians. Use with negative weight."""
    _, ee_quat = _ee_pose(env, asset_cfg)
    _, goal_quat = _goal_pose(env, command_name)
    result = quat_error_magnitude(ee_quat, goal_quat).clamp(max=3.14)
    _check_nan("orientation_error_l2", result, {"ee_quat": ee_quat, "goal_quat": goal_quat})
    return result


# =============================================================================
# Regularization (all clamped for safety)
# =============================================================================

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """||a_t - a_{t-1}||^2, clamped."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(delta**2, dim=-1).clamp(max=100.0)


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sum of squared joint velocities, clamped."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.joint_vel**2, dim=-1).clamp(max=100.0)


def base_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize base linear + angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin = torch.sum(asset.data.root_lin_vel_b**2, dim=-1)
    ang = torch.sum(asset.data.root_ang_vel_b**2, dim=-1)
    result = (lin + ang).clamp(max=100.0)
    _check_nan("base_velocity_penalty", result, {"lin_vel": asset.data.root_lin_vel_b, "ang_vel": asset.data.root_ang_vel_b})
    return result




def action_rate_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    arm_joint_names: list[str],
    base_joint_names: list[str],
    arm_weight: float = 1.0,
    base_weight: float = 1.0,
) -> torch.Tensor:
    """Action rate penalty with separate weights for arm and base joints.
    
    Returns weighted sum: arm_weight * ||Δa_arm||^2 + base_weight * ||Δa_base||^2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get joint indices
    arm_ids, _ = asset.find_joints(arm_joint_names)
    base_ids, _ = asset.find_joints(base_joint_names)
    
    # Get current and previous actions
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    # Compute action rate per joint group
    delta_action = action - prev_action
    
    # Note: action indices may not match joint indices directly
    # Assuming action order matches: [arm_actions, base_actions]
    num_arm = len(arm_ids)
    num_base = len(base_ids)
    
    # Split actions (assuming arm actions come first, then base)
    arm_delta = delta_action[:, :num_arm]
    base_delta = delta_action[:, num_arm:num_arm + num_base]
    
    # Weighted sum
    arm_penalty = arm_weight * torch.sum(arm_delta**2, dim=-1)
    base_penalty = base_weight * torch.sum(base_delta**2, dim=-1)
    
    result = arm_penalty + base_penalty
    _check_nan("action_rate_weighted", result, {"action": action, "prev_action": prev_action, "delta": delta_action})
    return result


def joint_vel_weighted(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    arm_joint_names: list[str],
    base_joint_names: list[str],
    arm_weight: float = 1.0,
    base_weight: float = 1.0,
) -> torch.Tensor:
    """Joint velocity penalty with separate weights for arm and base joints.
    
    Returns weighted sum: arm_weight * ||v_arm||^2 + base_weight * ||v_base||^2
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get joint indices
    arm_ids, _ = asset.find_joints(arm_joint_names)
    base_ids, _ = asset.find_joints(base_joint_names)
    
    # Get joint velocities
    joint_vel = asset.data.joint_vel
    
    # Compute velocity penalty per joint group
    arm_vel = joint_vel[:, arm_ids]
    base_vel = joint_vel[:, base_ids]
    
    # Weighted sum
    arm_penalty = arm_weight * torch.sum(arm_vel**2, dim=-1)
    base_penalty = base_weight * torch.sum(base_vel**2, dim=-1)
    
    result = arm_penalty + base_penalty
    _check_nan("joint_vel_weighted", result, {"joint_vel": joint_vel})
    return result


def base_approach_facing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    approach_threshold: float = 0.5,
    approach_sigma: float = 0.3,
) -> torch.Tensor:
    """R = facing * (1 + 1[d>d*] * approach)"""
    from isaaclab.utils.math import quat_apply

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # xy distance
    diff = cmd[:, :2] - asset.data.root_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)

    # facing = max(cos θ, 0)
    fwd = quat_apply(asset.data.root_quat_w,
                      torch.tensor([[1., 0., 0.]], device=env.device).expand(env.num_envs, -1))
    fwd_xy = fwd[:, :2] / fwd[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-3)
    dir_xy = diff / dist.unsqueeze(-1).clamp(min=1e-3)
    facing = torch.sum(fwd_xy * dir_xy, dim=-1).clamp(min=0.0)

    # approach = exp(-(d - d*) / σ), only when d > d*
    approach = torch.exp(-(dist - approach_threshold).clamp(min=0.0) / approach_sigma)
    far = (dist > approach_threshold).float()

    return facing * (1.0 + far * approach)


# =============================================================================
# Internal helpers
# =============================================================================

def _ee_and_goal(env, command_name, asset_cfg):
    """Extract current EE pose and goal pose (both world frame)."""
    asset = env.scene[asset_cfg.name]
    ee_state = asset.data.body_state_w[:, asset_cfg.body_ids[0], :]
    ee_pos = ee_state[:, :3]
    ee_quat = ee_state[:, 3:7]

    cmd = env.command_manager.get_command(command_name)
    goal_pos = cmd[:, :3]
    goal_quat = cmd[:, 3:7]

    return ee_pos, ee_quat, goal_pos, goal_quat


# =============================================================================
# Tracking rewards
# =============================================================================

def keypoint_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
    sigma: float = 0.15,
) -> torch.Tensor:
    """exp(-mean_kp_dist / sigma).  Range [0, 1]. 1 = perfect.

    Coarse shaping — provides gradient even far from target.
    """
    sigma = getattr(env, "_kp_sigma_exp", sigma)
    ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)
    _, mean_d = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)
    return torch.exp(-mean_d / sigma)


def keypoint_tracking_tanh(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
    sigma: float = 0.05,
) -> torch.Tensor:
    """1 - tanh(mean_kp_dist / sigma).  Range [0, 1].

    Fine-grained — steep gradient close to target.
    """
    sigma = getattr(env, "_kp_sigma_tanh", sigma)
    ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)
    _, mean_d = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)
    return 1.0 - torch.tanh(mean_d / sigma)


def keypoint_tracking_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
) -> torch.Tensor:
    """Mean keypoint L2 distance.  Use with negative weight."""
    ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)
    _, mean_d = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)
    return mean_d.clamp(max=5.0)


def keypoint_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
) -> torch.Tensor:
    ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)
    _, mean_d = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)

    if not hasattr(env, "_kp_best_dist"):
        # Initialize to current distance — first step progress = 0
        env._kp_best_dist = mean_d.clone()

    # Where best_dist is still the sentinel value, replace with current
    stale = env._kp_best_dist > 1e5
    env._kp_best_dist[stale] = mean_d[stale]

    progress = (env._kp_best_dist - mean_d).clamp(min=0.0)
    env._kp_best_dist = torch.min(env._kp_best_dist, mean_d)
    return progress


# =============================================================================
# Progress reset (call from EventTerm on reset)
# =============================================================================

def keypoint_progress_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Reset best-distance tracker for progress reward.

    Wire this as an EventTerm with mode="reset".
    """
    if hasattr(env, "_kp_best_dist"):
        env._kp_best_dist[env_ids] = 1e6


# =============================================================================
# Curriculum
# =============================================================================

# def keypoint_sigma_curriculum(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor,
#     command_name: str,
#     asset_cfg: SceneEntityCfg,
#     cube_side: float = 0.3,
# ):
#     stages = [
#         {"threshold": 0.08, "sigma_exp": 0.15,  "sigma_tanh": 0.05},
#         {"threshold": 0.04, "sigma_exp": 0.08,  "sigma_tanh": 0.025},
#         {"threshold": 0.02, "sigma_exp": 0.03,  "sigma_tanh": 0.01},
#         {"threshold": 0.008,"sigma_exp": 0.01,  "sigma_tanh": 0.004},
#         {"threshold": 0.003,"sigma_exp": 0.005, "sigma_tanh": 0.002},
#     ]

#     if not hasattr(env, "_kp_curriculum_stage"):
#         env._kp_curriculum_stage = 0
#         env._kp_sigma_exp = stages[0]["sigma_exp"]
#         env._kp_sigma_tanh = stages[0]["sigma_tanh"]
#         env._kp_curriculum_steps = 0
#         env._kp_curriculum_below_count = 0

#     env._kp_curriculum_steps += 1
#     stage = env._kp_curriculum_stage

#     # Min 1500 steps per stage
#     if env._kp_curriculum_steps < 1500 * (stage + 1):
#         return

#     # Already at max stage
#     if stage >= len(stages) - 1:
#         return

#     # Check median distance
#     ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)
#     _, mean_d = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)
#     avg_dist = mean_d.median().item()

#     # Consecutive confirmation
#     if avg_dist < stages[stage]["threshold"]:
#         env._kp_curriculum_below_count += 1
#     else:
#         env._kp_curriculum_below_count = 0

#     if env._kp_curriculum_below_count >= 100:
#         env._kp_curriculum_below_count = 0
#         stage += 1
#         env._kp_curriculum_stage = stage
#         env._kp_sigma_exp = stages[stage]["sigma_exp"]
#         env._kp_sigma_tanh = stages[stage]["sigma_tanh"]
#         print(f"[Curriculum] Stage {stage}: sigma_exp={stages[stage]['sigma_exp']}, "
#               f"sigma_tanh={stages[stage]['sigma_tanh']}, median_dist={avg_dist:.4f}m")

def keypoint_sigma_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
    arrival_threshold: float = 0.15,
    speed_threshold: float = 0.1,
):
    stages = [
        {"threshold": 0.08, "sigma_exp": 0.15,  "sigma_tanh": 0.05},
        {"threshold": 0.04, "sigma_exp": 0.08,  "sigma_tanh": 0.025},
        {"threshold": 0.02, "sigma_exp": 0.03,  "sigma_tanh": 0.01},
        {"threshold": 0.008,"sigma_exp": 0.01,  "sigma_tanh": 0.004},
        {"threshold": 0.003,"sigma_exp": 0.005, "sigma_tanh": 0.002},
    ]

    if not hasattr(env, "_kp_curriculum_stage"):
        env._kp_curriculum_stage = 0
        env._kp_sigma_exp = stages[0]["sigma_exp"]
        env._kp_sigma_tanh = stages[0]["sigma_tanh"]
        env._kp_curriculum_steps = 0
        env._kp_curriculum_below_count = 0
        # Per-env settled error tracking
        env._kp_settled_sum = torch.zeros(env.num_envs, device=env.device)
        env._kp_settled_count = torch.zeros(env.num_envs, device=env.device)

    env._kp_curriculum_steps += 1
    step = env._kp_curriculum_steps
    stage = env._kp_curriculum_stage
    debug = (step % 100 == 0)  # Print every 100 steps

    if stage >= len(stages) - 1:
        return

    # Current kp distance and EE speed
    ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)
    _, kp_dist = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)

    asset = env.scene[asset_cfg.name]
    ee_vel = asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10]
    ee_speed = torch.norm(ee_vel, dim=-1)

    # "Settled" = close enough AND slow enough
    close_enough = kp_dist < arrival_threshold
    slow_enough = ee_speed < speed_threshold
    settled = close_enough & slow_enough

    if debug:
        n = env.num_envs
        print(f"\n[Curriculum DEBUG] step={step} stage={stage}/{len(stages)-1}")
        print(f"  kp_dist:  mean={kp_dist.mean().item():.4f}m  median={kp_dist.median().item():.4f}m  "
              f"min={kp_dist.min().item():.4f}m  max={kp_dist.max().item():.4f}m")
        print(f"  ee_speed: mean={ee_speed.mean().item():.4f}m/s  max={ee_speed.max().item():.4f}m/s")
        print(f"  close_enough (<{arrival_threshold}m): {close_enough.sum().item()}/{n} ({100*close_enough.float().mean().item():.1f}%)")
        print(f"  slow_enough  (<{speed_threshold}m/s): {slow_enough.sum().item()}/{n} ({100*slow_enough.float().mean().item():.1f}%)")
        print(f"  settled (both):  {settled.sum().item()}/{n} ({100*settled.float().mean().item():.1f}%)")

    # Accumulate error only for settled envs
    env._kp_settled_sum[settled] += kp_dist[settled]
    env._kp_settled_count[settled] += 1

    # Reset on episode done (env_ids from curriculum manager = reset envs)
    if len(env_ids) > 0:
        env._kp_settled_sum[env_ids] = 0.0
        env._kp_settled_count[env_ids] = 0.0

    # Min steps guard
    min_steps = 1500 * (stage + 1)
    if step < min_steps:
        if debug:
            print(f"  BLOCKED: min_steps guard ({step}/{min_steps})")
        return

    # Need enough settled samples to make a decision
    has_data = env._kp_settled_count > 10  # at least 10 settled steps
    has_data_count = has_data.sum().item()
    need_count = env.num_envs * 0.3
    if has_data_count < need_count:
        if debug:
            print(f"  BLOCKED: not enough settled data. {has_data_count:.0f}/{need_count:.0f} envs have >10 settled steps")
            # Show distribution of settled counts
            sc = env._kp_settled_count
            print(f"  settled_count: mean={sc.mean().item():.1f}  max={sc.max().item():.0f}  "
                  f">0: {(sc>0).sum().item()}  >5: {(sc>5).sum().item()}  >10: {(sc>10).sum().item()}")
        return

    # Compute mean settled error per env, then take median across envs
    settled_mean = env._kp_settled_sum[has_data] / env._kp_settled_count[has_data]
    median_settled = settled_mean.median().item()
    threshold = stages[stage]["threshold"]

    if median_settled < threshold:
        env._kp_curriculum_below_count += 1
    else:
        env._kp_curriculum_below_count = 0

    if debug:
        print(f"  median_settled_error={median_settled:.4f}m  vs  threshold={threshold:.4f}m  "
              f"{'✓ BELOW' if median_settled < threshold else '✗ ABOVE'}")
        print(f"  below_count={env._kp_curriculum_below_count}/100")

    if env._kp_curriculum_below_count >= 100:
        env._kp_curriculum_below_count = 0
        stage += 1
        env._kp_curriculum_stage = stage
        env._kp_sigma_exp = stages[stage]["sigma_exp"]
        env._kp_sigma_tanh = stages[stage]["sigma_tanh"]
        # Reset settled tracking for fresh start at new stage
        env._kp_settled_sum.zero_()
        env._kp_settled_count.zero_()
        print(f"\n{'='*60}")
        print(f"[Curriculum] ▶ Stage {stage}: sigma_exp={stages[stage]['sigma_exp']}, "
              f"sigma_tanh={stages[stage]['sigma_tanh']}, median_settled={median_settled:.4f}m")
        print(f"{'='*60}\n")
              
def keypoint_settle(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
    sigma_d: float = 0.05,
    sigma_v: float = 0.05,
) -> torch.Tensor:
    """R = exp(-d_kp / sigma_d) * exp(-||v_ee|| / sigma_v)

    Unique global maximum at (d=0, v=0). Continuous, differentiable.
    sigma_d: distance gate width, ~0.05-0.1m. Controls how early
             the "slow down" signal kicks in during approach.
    sigma_v: velocity tolerance, ~0.05 m/s. Controls how aggressively
             residual velocity is penalized near target.
    """
    ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)
    _, mean_d = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)

    asset = env.scene[asset_cfg.name]
    ee_vel = asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10]
    ee_speed = torch.norm(ee_vel, dim=-1)

    proximity = torch.exp(-mean_d / sigma_d)
    stillness = torch.exp(-ee_speed / sigma_v)

    return proximity * stillness


def keypoint_metrics(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
) -> torch.Tensor:
    """Compute and log tracking metrics. Returns 0 (use weight=0)."""
    from isaaclab.utils.math import quat_error_magnitude

    ee_p, ee_q, g_p, g_q = _ee_and_goal(env, command_name, asset_cfg)

    # Keypoint distance
    kp_dists, mean_kp = keypoint_distance(ee_p, ee_q, g_p, g_q, cube_side)

    # Pure position error
    pos_err = torch.norm(ee_p - g_p, dim=-1)

    # Pure orientation error
    ori_err = quat_error_magnitude(ee_q, g_q)

    # Curriculum stage
    stage = getattr(env, "_kp_curriculum_stage", 0)

    # Store in extras — rsl_rl picks these up automatically
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Metrics/kp_mean_dist_cm"] = mean_kp.mean().item() * 100
    env.extras["log"]["Metrics/pos_error_cm"] = pos_err.mean().item() * 100
    env.extras["log"]["Metrics/ori_error_deg"] = torch.rad2deg(ori_err).mean().item()
    env.extras["log"]["Metrics/curriculum_stage"] = float(stage)

    # Percentile stats (useful to see distribution)
    env.extras["log"]["Metrics/pos_error_median_cm"] = pos_err.median().item() * 100
    env.extras["log"]["Metrics/pos_error_90pct_cm"] = torch.quantile(pos_err, 0.9).item() * 100
    env.extras["log"]["Metrics/ori_error_median_deg"] = torch.rad2deg(ori_err.median()).item()

    return torch.zeros(env.num_envs, device=env.device)