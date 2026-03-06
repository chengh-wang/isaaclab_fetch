# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential drive action for wheeled mobile robots."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialDriveAction(ActionTerm):
    """Action term for differential drive mobile base control.
    
    This action term converts 2D velocity commands (v_x, ω_z) into wheel velocity
    targets using differential drive kinematics. The robot base moves forward/backward
    based on linear velocity and rotates based on angular velocity.
    
    Kinematic equations:
        ω_left  = (v_x - ω_z * wheel_separation/2) / wheel_radius
        ω_right = (v_x + ω_z * wheel_separation/2) / wheel_radius
    
    The action dimension is 2: [linear_velocity, angular_velocity]
    """
    
    cfg: "DifferentialDriveActionCfg"
    """The configuration of the action term."""
    
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    
    def __init__(self, cfg: "DifferentialDriveActionCfg", env: ManagerBasedEnv):
        """Initialize the differential drive action term.
        
        Args:
            cfg: The configuration for the action term.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # Find wheel joint indices
        left_joint_ids, left_joint_names = self._asset.find_joints(cfg.left_wheel_joint_name)
        if len(left_joint_ids) != 1:
            raise ValueError(
                f"Expected exactly 1 left wheel joint matching '{cfg.left_wheel_joint_name}', "
                f"found {len(left_joint_ids)}: {left_joint_names}"
            )
        
        right_joint_ids, right_joint_names = self._asset.find_joints(cfg.right_wheel_joint_name)
        if len(right_joint_ids) != 1:
            raise ValueError(
                f"Expected exactly 1 right wheel joint matching '{cfg.right_wheel_joint_name}', "
                f"found {len(right_joint_ids)}: {right_joint_names}"
            )
        
        self._left_wheel_idx = left_joint_ids[0]
        self._right_wheel_idx = right_joint_ids[0]
        self._joint_ids = [self._left_wheel_idx, self._right_wheel_idx]
        
        # Store kinematic parameters
        self._wheel_radius = cfg.wheel_radius
        self._wheel_separation = cfg.wheel_separation
        self._linear_scale = cfg.linear_velocity_scale
        self._angular_scale = cfg.angular_velocity_scale
        self._max_linear = cfg.max_linear_velocity
        self._max_angular = cfg.max_angular_velocity
        
        # Create action tensors
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._wheel_velocities = torch.zeros(self.num_envs, 2, device=self.device)
        
        print(f"[DifferentialDriveAction] Initialized with:")
        print(f"  Left wheel joint: {left_joint_names[0]} (idx: {self._left_wheel_idx})")
        print(f"  Right wheel joint: {right_joint_names[0]} (idx: {self._right_wheel_idx})")
        print(f"  Wheel radius: {self._wheel_radius} m")
        print(f"  Wheel separation: {self._wheel_separation} m")
    
    @property
    def action_dim(self) -> int:
        """Dimension of the action space (linear_vel, angular_vel)."""
        return 2
    
    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions received from the policy."""
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions after scaling and clipping."""
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        """Process the incoming velocity commands.
        
        Args:
            actions: Tensor of shape (num_envs, 2) with [linear_vel, angular_vel].
        """
        # Store raw actions
        self._raw_actions[:] = actions
        
        # Scale actions
        self._processed_actions[:, 0] = actions[:, 0] * self._linear_scale
        self._processed_actions[:, 1] = actions[:, 1] * self._angular_scale
        
        # Clip to velocity limits
        self._processed_actions[:, 0] = torch.clamp(
            self._processed_actions[:, 0], 
            -self._max_linear, 
            self._max_linear
        )
        self._processed_actions[:, 1] = torch.clamp(
            self._processed_actions[:, 1], 
            -self._max_angular, 
            self._max_angular
        )
    
    def apply_actions(self):
        v_x = self._processed_actions[:, 0]
        omega_z = self._processed_actions[:, 1]
        
        half_separation = self._wheel_separation / 2.0
        self._wheel_velocities[:, 0] = (v_x - omega_z * half_separation) / self._wheel_radius
        self._wheel_velocities[:, 1] = (v_x + omega_z * half_separation) / self._wheel_radius
        
        self._asset.set_joint_velocity_target(
            self._wheel_velocities, 
            joint_ids=self._joint_ids
        )
        
        # ===== DEBUG =====
        if self._env.common_step_counter % 10 == 0:
            # What policy commanded
            raw = self._raw_actions[0].cpu().tolist()
            proc = self._processed_actions[0].cpu().tolist()    
            wheel_tgt = self._wheel_velocities[0].cpu().tolist()
            
            # What actually happened
            actual_vel = self._asset.data.joint_vel[0, self._joint_ids].cpu().tolist()
            actual_effort = self._asset.data.applied_torque[0, self._joint_ids].cpu().tolist()
            
            # Base velocity
            base_lin = self._asset.data.root_lin_vel_b[0].cpu().tolist()
            base_ang = self._asset.data.root_ang_vel_b[0].cpu().tolist()
            
            # print(f"\n[DiffDrive DEBUG] step={self._env.common_step_counter}")
            # print(f"  raw_action:     [{raw[0]:+.4f}, {raw[1]:+.4f}]  (v_x, omega_z)")
            # print(f"  processed:      [{proc[0]:+.4f}, {proc[1]:+.4f}]")
            # print(f"  wheel_vel_tgt:  L={wheel_tgt[0]:+.4f}  R={wheel_tgt[1]:+.4f} rad/s")
            # print(f"  wheel_vel_act:  L={actual_vel[0]:+.4f}  R={actual_vel[1]:+.4f} rad/s")
            # print(f"  wheel_torque:   L={actual_effort[0]:+.4f}  R={actual_effort[1]:+.4f} Nm")
            # print(f"  base_lin_vel:   x={base_lin[0]:+.4f} y={base_lin[1]:+.4f} z={base_lin[2]:+.4f}")
            # print(f"  base_ang_vel:   x={base_ang[0]:+.4f} y={base_ang[1]:+.4f} z={base_ang[2]:+.4f}")

            # # Ground contact forces
            # sensor = None
            # if "contact_sensor" in self._env.scene.keys():
            #     sensor = self._env.scene["contact_sensor"]
            # elif hasattr(self._env.scene, "contact_sensor"):
            #     sensor = getattr(self._env.scene, "contact_sensor")

            # if sensor is not None:
            #     forces = sensor.data.net_forces_w[0]
            #     print(f"\n  Ground Contact Forces:")
            #     print(f"  {'link':<35s} {'z_pos':>8s} {'Fx':>8s} {'Fy':>8s} {'Fz':>10s}")
            #     print(f"  {'-'*70}")
            #     for i, name in enumerate(self._asset.body_names):
            #         fx = forces[i, 0].item()
            #         fy = forces[i, 1].item()
            #         fz = forces[i, 2].item()
            #         z_pos = self._asset.data.body_state_w[0, i, 2].item()
            #         if abs(fz) > 0.1 or abs(fx) > 0.1 or abs(fy) > 0.1 or "wheel" in name or "caster" in name or name == "base_link":
            #             print(f"  {name:<35s} {z_pos:>8.4f} {fx:>8.2f} {fy:>8.2f} {fz:>10.2f} N")
            # else:
            #     # No contact sensor - just print z positions
            #     print(f"\n  Link heights (no ContactSensor):")
            #     for i, name in enumerate(self._asset.body_names):
            #         z = self._asset.data.body_state_w[0, i, 2].item()
            #         if z < 0.15 or "wheel" in name or "caster" in name or name == "base_link":
            #             print(f"  {name:<35s} z={z:.4f}")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        
        Args:
            env_ids: The environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._wheel_velocities[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._wheel_velocities[env_ids] = 0.0


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Configuration for differential drive velocity action.
    
    This action takes 2D velocity commands (linear_x, angular_z) and converts them
    to wheel velocity targets using differential drive kinematics.
    """
    
    class_type: type[ActionTerm] = DifferentialDriveAction
    """The class type for the action term."""
    
    asset_name: str = "robot"
    """Name of the asset in the environment."""
    
    left_wheel_joint_name: str = "l_wheel_joint"
    """Name of the left wheel joint."""
    
    right_wheel_joint_name: str = "r_wheel_joint"
    """Name of the right wheel joint."""
    
    wheel_radius: float = 0.0625
    """Radius of the wheels in meters. Default is Fetch robot wheel radius."""
    
    wheel_separation: float = 0.372
    """Distance between the wheels (track width) in meters. Default is Fetch robot."""
    
    linear_velocity_scale: float = 1.0
    """Scale factor for linear velocity input."""
    
    angular_velocity_scale: float = 1.0
    """Scale factor for angular velocity input."""
    
    max_linear_velocity: float = 1.0
    """Maximum linear velocity in m/s. Real Fetch: 1.0 m/s."""
    
    max_angular_velocity: float = 1.57
    """Maximum angular velocity in rad/s. Real Fetch: ~1.57 rad/s (90 deg/s)."""
