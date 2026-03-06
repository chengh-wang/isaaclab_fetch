"""Custom termination conditions for Fetch robot environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_flipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt_angle: float = 0.5,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    cos_max_tilt = torch.cos(torch.tensor(max_tilt_angle, device=asset.device))
    
    # DEBUG: uncomment to see actual values
    # if env.common_step_counter % 100 == 0:
    #     print(f"gravity_z: min={projected_gravity[:, 2].min():.3f}, "
    #           f"max={projected_gravity[:, 2].max():.3f}, "
    #           f"threshold={-cos_max_tilt:.3f}, "
    #           f"num_flipped={( projected_gravity[:, 2] > -cos_max_tilt).sum()}")
    
    return projected_gravity[:, 2] > -cos_max_tilt


def base_height_too_low(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.3,
) -> torch.Tensor:
    """Terminate if base drops below threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < min_height


def reset_rfm_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset RFM state for specified environments.
    
    This should be called as a reset event to initialize RFM state
    (epsilon_0, cumulative error, etc.) at episode start.
    
    Args:
        env: The environment instance.
        env_ids: Environment indices being reset.
        command_name: Name of the pose command.
        asset_cfg: Configuration for the robot asset.
    """
    from .rfm_rewards import get_rfm_state, compute_se3_distance, RFM_CFG, get_ee_pose, get_goal_pose
    
    if len(env_ids) == 0:
        return
    
    rfm = get_rfm_state(env)
    
    # Get current EE pose and goal pose
    ee_pos, ee_quat = get_ee_pose(env, asset_cfg)
    goal_pos, goal_quat = get_goal_pose(env, command_name)
    
    # Compute initial SE(3) distance
    epsilon_0 = compute_se3_distance(
        ee_pos[env_ids], ee_quat[env_ids],
        goal_pos[env_ids], goal_quat[env_ids],
        RFM_CFG.w_p, RFM_CFG.w_o
    )
    
    # Reset RFM state for these environments
    rfm.reset(env_ids, epsilon_0)

