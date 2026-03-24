"""Observation delay wrapper for sim2real robustness.

Provides DelayedObsTerm (generic delay buffer) plus pre-configured
delayed observation functions with explicit signatures for IsaacLab's
ObservationManager, which parses function signatures.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Import observation functions we wrap
from .observations import base_lin_vel, base_ang_vel, keypoint_command_in_body_frame
from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel


# =============================================================================
# Generic delay buffer
# =============================================================================

class DelayedObsTerm:
    """Wraps any observation function with a configurable delay (in seconds).

    - Delay is per-env randomized within [min_delay_s, max_delay_s]
    - On first call, converts seconds to discrete steps using env.step_dt
    - On reset, re-randomizes delays for reset envs
    """

    def __init__(self, func, min_delay_s: float = 0.0, max_delay_s: float = 0.1):
        self.func = func
        self.min_delay_s = min_delay_s
        self.max_delay_s = max_delay_s
        # Initialized lazily on first __call__
        self.buffer = None
        self.delays = None
        self.step = 0
        self._dt = None
        self._min_steps = None
        self._max_steps = None
        self._pending_reset_ids = None

    def _lazy_init(self, env, B: int, D: int, device: torch.device):
        """Convert seconds -> steps on first call, allocate buffer."""
        self._dt = env.step_dt  # = sim_dt * decimation
        self._min_steps = max(0, int(self.min_delay_s / self._dt))
        self._max_steps = max(self._min_steps, int(self.max_delay_s / self._dt))

        buf_len = self._max_steps + 1
        self.buffer = torch.zeros(buf_len, B, D, device=device)

        if self._min_steps == self._max_steps:
            self.delays = torch.full((B,), self._min_steps, device=device, dtype=torch.long)
        else:
            self.delays = torch.randint(
                self._min_steps, self._max_steps + 1, (B,), device=device
            )

    def __call__(self, env, **kwargs) -> torch.Tensor:
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        raw = self.func(env, **filtered)
        B, D = raw.shape

        if self.buffer is None:
            self._lazy_init(env, B, D, raw.device)
            # Fill entire buffer with first observation (no stale data)
            self.buffer[:] = raw.unsqueeze(0)

        # Handle pending resets: fill buffer with current real obs (not zeros)
        if self._pending_reset_ids is not None:
            ids = self._pending_reset_ids
            for i in range(self.buffer.shape[0]):
                self.buffer[i, ids] = raw[ids]
            self._pending_reset_ids = None

        buf_len = self._max_steps + 1

        # Write current observation
        write_idx = self.step % buf_len
        self.buffer[write_idx] = raw

        # Read delayed observation (per-env)
        read_idx = (self.step - self.delays) % buf_len
        delayed = self.buffer[read_idx, torch.arange(B, device=raw.device)]

        self.step += 1
        return delayed

    def reset(self, env_ids: torch.Tensor):
        """Re-randomize delays; buffer will be filled with real obs on next __call__."""
        if self.delays is None:
            return
        if self._min_steps == self._max_steps:
            self.delays[env_ids] = self._min_steps
        else:
            self.delays[env_ids] = torch.randint(
                self._min_steps, self._max_steps + 1,
                (len(env_ids),), device=self.delays.device,
            )
        # Defer buffer fill to next __call__ where we have real obs
        self._pending_reset_ids = env_ids


# =============================================================================
# Pre-configured delay instances
# =============================================================================

# Joint encoder: RS-485 roundtrip ~ 5-15ms
_delayed_joint_pos = DelayedObsTerm(joint_pos_rel, min_delay_s=0.0, max_delay_s=0.025)

# Joint velocity: same RS-485 delay as position
_delayed_joint_vel = DelayedObsTerm(joint_vel_rel, min_delay_s=0.0, max_delay_s=0.025)

# Keypoint: FK computed from delayed joint pos
_delayed_kp_cmd = DelayedObsTerm(keypoint_command_in_body_frame, min_delay_s=0.0, max_delay_s=0.025)

# Base IMU + odometry fusion ~ 10-20ms
_delayed_base_lin = DelayedObsTerm(base_lin_vel, min_delay_s=0.0, max_delay_s=0.020)
_delayed_base_ang = DelayedObsTerm(base_ang_vel, min_delay_s=0.0, max_delay_s=0.020)

ALL_DELAYED_WRAPPERS = [_delayed_joint_pos, _delayed_joint_vel, _delayed_kp_cmd, _delayed_base_lin, _delayed_base_ang]


# =============================================================================
# Explicit-signature wrappers (IsaacLab ObservationManager parses signatures)
# =============================================================================

def delayed_joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _delayed_joint_pos(env, asset_cfg=asset_cfg)

def delayed_joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _delayed_joint_vel(env, asset_cfg=asset_cfg)

def delayed_kp_cmd(
    env: ManagerBasedEnv,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_side: float = 0.3,
) -> torch.Tensor:
    return _delayed_kp_cmd(env, command_name=command_name, asset_cfg=asset_cfg, cube_side=cube_side)

def delayed_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _delayed_base_lin(env, asset_cfg=asset_cfg)

def delayed_base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _delayed_base_ang(env, asset_cfg=asset_cfg)


# =============================================================================
# Reset helper for EventManager
# =============================================================================

def reset_obs_delay_buffers(env: ManagerBasedEnv, env_ids: torch.Tensor | None):
    """Reset delay buffers for all delayed observation wrappers."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for wrapper in ALL_DELAYED_WRAPPERS:
        wrapper.reset(env_ids)
