# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for Fetch robot environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul
from .utils import pose_to_keypoints, keypoint_distance, keypoint_delta_in_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the body frame.
    
    Returns:
        Linear velocity (x, y, z) in the robot's body frame. Shape: (num_envs, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the body frame.
    
    Returns:
        Angular velocity (roll_rate, pitch_rate, yaw_rate) in the robot's body frame. Shape: (num_envs, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def pose_command_in_body_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Transform the world-frame pose command to the robot's body frame.
    
    This observation gives the target end-effector pose relative to the robot's
    current base pose, which is invariant to the robot's global position/orientation.
    
    The command is expected to contain: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    
    Returns:
        Pose command in body frame: [rel_pos_x, rel_pos_y, rel_pos_z, rel_quat_w, rel_quat_x, rel_quat_y, rel_quat_z]
        Shape: (num_envs, 7)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the command (world frame pose)
    command = env.command_manager.get_command(command_name)
    
    # Command format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    target_pos_w = command[:, :3]  # (num_envs, 3)
    target_quat_w = command[:, 3:7]  # (num_envs, 4) - [w, x, y, z]
    
    # Get robot base pose in world frame
    base_pos_w = asset.data.root_pos_w  # (num_envs, 3)
    base_quat_w = asset.data.root_quat_w  # (num_envs, 4) - [w, x, y, z]
    
    # Transform target position to body frame
    # rel_pos_b = R_base^(-1) @ (target_pos_w - base_pos_w)
    rel_pos_w = target_pos_w - base_pos_w
    rel_pos_b = quat_apply_inverse(base_quat_w, rel_pos_w)
    
    # Transform target orientation to body frame
    # rel_quat_b = quat_conjugate(base_quat_w) * target_quat_w
    base_quat_conj = quat_conjugate(base_quat_w)
    rel_quat_b = quat_mul(base_quat_conj, target_quat_w)
    
    return torch.cat([rel_pos_b, rel_quat_b], dim=-1)


def pose_command_position_in_body_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Transform just the position part of the world-frame pose command to the robot's body frame.
    
    Returns:
        Position command in body frame: [rel_pos_x, rel_pos_y, rel_pos_z]
        Shape: (num_envs, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the command (world frame pose)
    command = env.command_manager.get_command(command_name)
    
    # Command format: [pos_x, pos_y, pos_z, ...]
    target_pos_w = command[:, :3]  # (num_envs, 3)
    
    # Get robot base pose in world frame
    base_pos_w = asset.data.root_pos_w  # (num_envs, 3)
    base_quat_w = asset.data.root_quat_w  # (num_envs, 4)
    
    # Transform target position to body frame
    rel_pos_w = target_pos_w - base_pos_w
    rel_pos_b = quat_apply_inverse(base_quat_w, rel_pos_w)
    
    return rel_pos_b


# =============================================================================
# RFM Observations
# =============================================================================

def rfm_se3_distance_ref(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SE(3) distance reference observation for RFM.
    
    This observation:
    1. Computes current SE(3) distance to target
    2. Updates RFM state manager (step)
    3. Returns epsilon_ref for policy input
    
    Should be called once per step as an observation.
    
    Returns:
        epsilon_ref: Reference SE(3) distance based on expected approach. Shape: (num_envs, 1)
    """
    from .rfm_rewards import get_rfm_state, compute_se3_distance, RFM_CFG, get_ee_pose, get_goal_pose
    
    rfm = get_rfm_state(env)
    
    # Get current poses
    ee_pos, ee_quat = get_ee_pose(env, asset_cfg)
    goal_pos, goal_quat = get_goal_pose(env, command_name)
    
    # Compute current SE(3) distance
    epsilon_t = compute_se3_distance(ee_pos, ee_quat, goal_pos, goal_quat,
                                      RFM_CFG.w_p, RFM_CFG.w_o)
    
    # Step RFM state (updates D_t, cumulative error, etc.)
    dt = env.step_dt
    rfm.step(epsilon_t, dt)
    
    return rfm.epsilon_ref.unsqueeze(-1)


def rfm_phase_variable(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Phase variable D_t for debugging/visualization.
    
    D_t = 0: manipulation mode (target is close)
    D_t = 1: locomotion mode (target is far)
    
    Returns:
        D_t: Phase variable. Shape: (num_envs, 1)
    """
    from .rfm_rewards import get_rfm_state
    
    rfm = get_rfm_state(env)
    return rfm.D_t.unsqueeze(-1)


def keypoint_command_in_body_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cube_side: float = 0.3,
) -> torch.Tensor:
    """Observation: keypoint delta (goal - current) in robot base frame.

    Returns:
        (N, 9) — 3 keypoints x 3D, continuous and smooth.

    Replaces the old pose_command_in_body_frame (7-dim pos+quat delta).
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    # Current EE pose (world frame)
    ee_state = asset.data.body_state_w[:, asset_cfg.body_ids[0], :]
    ee_pos_w = ee_state[:, :3]
    ee_quat_w = ee_state[:, 3:7]

    # Goal pose (world frame, from command)
    goal_pos_w = cmd[:, :3]
    goal_quat_w = cmd[:, 3:7]

    # Base frame quaternion
    base_quat_w = asset.data.root_quat_w  # (N, 4)

    return keypoint_delta_in_frame(
        curr_pos=ee_pos_w,
        curr_quat=ee_quat_w,
        goal_pos=goal_pos_w,
        goal_quat=goal_quat_w,
        frame_quat=base_quat_w,
        cube_side=cube_side,
    )






