# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame pose command generator for mobile manipulation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class WorldPoseCommand(CommandTerm):
    """Command generator for world-frame pose targets.
    
    Generates target poses in world frame relative to each environment's origin.
    The command is a 7D vector: [x, y, z, quat_w, quat_x, quat_y, quat_z].
    
    Features:
    - Success-based resampling: new target when EE reaches current target
    - Strict success criteria: position + orientation + settling speed
    - Configurable hold time at target before resampling
    - RFM state reset on new target
    """
    
    cfg: "WorldPoseCommandCfg"
    
    def __init__(self, cfg: "WorldPoseCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get the robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Get body index for visualization
        self._body_idx = self._asset.find_bodies(cfg.body_name)[0][0]
        
        # Get environment origins for offset
        self._env_origins = env.scene.env_origins
        
        # Create command buffer: [x, y, z, quat_w, quat_x, quat_y, quat_z]
        self._command = torch.zeros(self.num_envs, 7, device=self.device)
        
        # Store ranges
        self._pos_x_range = cfg.ranges.pos_x
        self._pos_y_range = cfg.ranges.pos_y
        self._pos_z_range = cfg.ranges.pos_z
        self._roll_range = cfg.ranges.roll
        self._pitch_range = cfg.ranges.pitch
        self._yaw_range = cfg.ranges.yaw
        
        # Success criteria thresholds
        self._success_threshold = cfg.success_threshold
        self._ori_threshold = cfg.ori_threshold
        self._settling_speed_threshold = cfg.settling_speed_threshold
        
        # Hold time range (seconds to wait at target before resampling)
        self._hold_time_range = cfg.hold_time_range
        
        # Track which envs just reached target (for RFM reset)
        self._just_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Track hold state for each env
        self._is_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_start_time = torch.zeros(self.num_envs, device=self.device)
        self._hold_duration = torch.zeros(self.num_envs, device=self.device)
        self._env_time = torch.zeros(self.num_envs, device=self.device)
        
        # Metrics (logged to WandB at episode end)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ori"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ee_target_distance"] = torch.zeros(self.num_envs, device=self.device)
        
    def __str__(self) -> str:
        return f"WorldPoseCommand(device={self.device})"
    
    @property
    def command(self) -> torch.Tensor:
        """The current pose command in world frame. Shape: (num_envs, 7)."""
        return self._command
    
    @property
    def just_reached_target(self) -> torch.Tensor:
        """Boolean mask of envs that just reached target this step."""
        return self._just_reached
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for specified environments."""
        n = len(env_ids)
        if n == 0:
            return
        
        # Sample positions in world frame (relative to env origin)
        pos_x = torch.empty(n, device=self.device).uniform_(*self._pos_x_range)
        pos_y = torch.empty(n, device=self.device).uniform_(*self._pos_y_range)
        pos_z = torch.empty(n, device=self.device).uniform_(*self._pos_z_range)
        
        # Add environment origin offset (world frame)
        self._command[env_ids, 0] = self._env_origins[env_ids, 0] + pos_x
        self._command[env_ids, 1] = self._env_origins[env_ids, 1] + pos_y
        self._command[env_ids, 2] = pos_z  # z is absolute, not env-relative
        
        # Sample orientation
        roll = torch.empty(n, device=self.device).uniform_(*self._roll_range)
        pitch = torch.empty(n, device=self.device).uniform_(*self._pitch_range)
        yaw = torch.empty(n, device=self.device).uniform_(*self._yaw_range)
        
        # Convert to quaternion
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        self._command[env_ids, 3:7] = quat
        
        # Reset RFM state for these environments
        self._reset_rfm_for_envs(env_ids)
    
    def _reset_rfm_for_envs(self, env_ids: Sequence[int]):
        """Reset RFM state when command is resampled."""
        try:
            from .rfm_rewards import get_rfm_state, compute_se3_distance, RFM_CFG
            
            rfm = get_rfm_state(self._env)
            
            # Get current EE pose
            body_pose = self._asset.data.body_state_w[:, self._body_idx]
            ee_pos = body_pose[:, :3]
            ee_quat = body_pose[:, 3:7]
            
            # Get new goal pose
            goal_pos = self._command[:, :3]
            goal_quat = self._command[:, 3:7]
            
            # Compute initial SE(3) distance
            epsilon_0 = compute_se3_distance(
                ee_pos[env_ids], ee_quat[env_ids],
                goal_pos[env_ids], goal_quat[env_ids],
                RFM_CFG.w_p, RFM_CFG.w_o
            )
            
            # Reset RFM state for these environments
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            rfm.reset(env_ids_tensor, epsilon_0)
        except ImportError:
            pass  # RFM not available
    
    def _update_command(self):
        """Check for success and resample after hold time at target.
        
        Success requires ALL of the following:
        1. Position error < success_threshold
        2. Orientation error < ori_threshold
        3. EE linear speed < settling_speed_threshold
        """
        # Clear previous step's flag
        self._just_reached.fill_(False)
        
        # Update environment time
        dt = self._env.step_dt
        self._env_time += dt
        
        # Get current EE state
        body_pose = self._asset.data.body_state_w[:, self._body_idx]
        ee_pos = body_pose[:, :3]
        ee_quat = body_pose[:, 3:7]
        ee_vel = body_pose[:, 7:10]  # linear velocity
        
        # --- Criterion 1: Position error ---
        target_pos = self._command[:, :3]
        pos_distance = torch.norm(ee_pos - target_pos, dim=-1)
        pos_ok = pos_distance < self._success_threshold
        
        # --- Criterion 2: Orientation error ---
        target_quat = self._command[:, 3:7]
        ori_error = quat_error_magnitude(ee_quat, target_quat)
        ori_ok = ori_error < self._ori_threshold
        
        # --- Criterion 3: EE settling speed ---
        ee_speed = torch.norm(ee_vel, dim=-1)
        speed_ok = ee_speed < self._settling_speed_threshold
        
        # --- Combined success criterion ---
        reached = pos_ok & ori_ok & speed_ok
        
        # Check for envs that just satisfied all criteria (not already holding)
        just_reached = reached & (~self._is_holding)
        just_reached_ids = torch.where(just_reached)[0]
        
        if len(just_reached_ids) > 0:
            # Start hold timer for these envs
            self._is_holding[just_reached_ids] = True
            self._hold_start_time[just_reached_ids] = self._env_time[just_reached_ids]
            # Sample random hold duration for each env
            self._hold_duration[just_reached_ids] = torch.empty(
                len(just_reached_ids), device=self.device
            ).uniform_(*self._hold_time_range)
        
        # Cancel hold if env drifts out of success zone during hold
        drifted = self._is_holding & (~reached)
        self._is_holding[drifted] = False
        
        # Check for envs where hold time has expired (still in success zone)
        hold_elapsed = self._env_time - self._hold_start_time
        hold_expired = self._is_holding & (hold_elapsed >= self._hold_duration)
        expired_ids = torch.where(hold_expired)[0]
        
        if len(expired_ids) > 0:
            self._just_reached[expired_ids] = True
            self._is_holding[expired_ids] = False
            self._resample_command(expired_ids)
    
    def _update_metrics(self):
        """Update metrics based on current state."""
        # Get current EE pose
        body_pose = self._asset.data.body_state_w[:, self._body_idx]
        ee_pos = body_pose[:, :3]
        ee_quat = body_pose[:, 3:7]
        
        # Get target pose
        target_pos = self._command[:, :3]
        target_quat = self._command[:, 3:7]
        
        # Position error (L2 norm)
        self.metrics["error_pos"] = torch.norm(ee_pos - target_pos, dim=-1)
        
        # Orientation error (shortest path in radians)
        self.metrics["error_ori"] = quat_error_magnitude(ee_quat, target_quat)
        
        # Total distance (for convenience)
        self.metrics["ee_target_distance"] = self.metrics["error_pos"]
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif len(env_ids) == 0:
            return {}
        
        # Reset hold state for these envs
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._is_holding[env_ids_tensor] = False
        self._env_time[env_ids_tensor] = 0.0
        
        self._resample_command(env_ids)
        return {}
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_marker"):
                # Create goal pose marker (frame axes)
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                self.goal_marker = VisualizationMarkers(marker_cfg)
                
                # Create current pose marker
                current_marker_cfg = FRAME_MARKER_CFG.copy()
                current_marker_cfg.prim_path = "/Visuals/Command/current_pose"
                current_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
                self.current_marker = VisualizationMarkers(current_marker_cfg)
            
            self.goal_marker.set_visibility(True)
            self.current_marker.set_visibility(True)
        else:
            if hasattr(self, "goal_marker"):
                self.goal_marker.set_visibility(False)
                self.current_marker.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not self._asset.is_initialized:
            return
        
        # Visualize goal pose
        self.goal_marker.visualize(self._command[:, :3], self._command[:, 3:7])
        
        # Visualize current end-effector pose
        body_pose = self._asset.data.body_link_pose_w[:, self._body_idx]
        self.current_marker.visualize(body_pose[:, :3], body_pose[:, 3:7])


@configclass
class WorldPoseCommandCfg(CommandTermCfg):
    """Configuration for world-frame pose command generator.
    
    Success criteria (ALL must be satisfied simultaneously):
    - Position error < success_threshold
    - Orientation error < ori_threshold  
    - EE linear speed < settling_speed_threshold
    
    After all criteria are met, the robot must hold for a random duration
    sampled from hold_time_range. If it drifts out during hold, the hold
    timer resets.
    """
    
    class_type: type = WorldPoseCommand
    """The class type for the command term."""
    
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    body_name: str = "wrist_roll_link"
    """Name of the end-effector body for visualization."""
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for pose commands in world frame."""
        pos_x: tuple[float, float] = (0.3, 0.8)
        """Range for x position (world frame, relative to env origin)."""
        pos_y: tuple[float, float] = (-0.5, 0.5)
        """Range for y position (world frame, relative to env origin)."""
        pos_z: tuple[float, float] = (0.5, 1.2)
        """Range for z position (world frame, absolute)."""
        roll: tuple[float, float] = (0.0, 0.0)
        """Range for roll angle."""
        pitch: tuple[float, float] = (0.0, 0.0)
        """Range for pitch angle."""
        yaw: tuple[float, float] = (-3.14, 3.14)
        """Range for yaw angle."""
    
    ranges: Ranges = Ranges()
    """Ranges for the pose command sampling."""
    
    # ---------- Success criteria ----------
    success_threshold: float = 0.1
    """Position distance threshold (meters) for considering target reached."""
    
    ori_threshold: float = 0.3
    """Orientation error threshold (radians) for considering target reached.
    This is the shortest-path quaternion error magnitude.
    ~0.3 rad ≈ 17 deg. Use 0.15 for ~8.5 deg (very strict)."""
    
    settling_speed_threshold: float = 0.1
    """EE linear speed threshold (m/s). Must be below this to count as 'settled'.
    Prevents counting a fast pass-through as success."""
    
    # ---------- Hold time ----------
    hold_time_range: tuple[float, float] = (0.0, 5.0)
    """Range (min, max) in seconds to hold at target before resampling.
    Sampled uniformly for each env when target is first reached.
    If the EE drifts out of the success zone during hold, the hold is cancelled
    and the robot must re-enter the success zone to restart the hold timer."""