"""Observation delay modifier for sim2real robustness.

Uses IsaacLab's ModifierBase so ObservationManager handles reset automatically.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.utils.modifiers import ModifierBase, ModifierCfg


class ObsDelayModifier(ModifierBase):
    """Delays observations by a random number of steps per env.

    Applied as a modifier on ObsTerm — no wrapper functions or reset events needed.
    ObservationManager calls reset() automatically on env reset.
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str):
        super().__init__(cfg, data_dim, device)
        self.min_steps: int = cfg.params.get("min_delay_steps", 0)
        self.max_steps: int = cfg.params.get("max_delay_steps", 3)
        # Clear params so they aren't passed to __call__ by ObservationManager
        cfg.params = {}

        buf_len = self.max_steps + 1
        B = data_dim[0]
        D = data_dim[1] if len(data_dim) > 1 else 1

        self.buffer = torch.zeros(buf_len, B, D, device=device)
        if self.min_steps == self.max_steps:
            self.delays = torch.full((B,), self.min_steps, device=device, dtype=torch.long)
        else:
            self.delays = torch.randint(self.min_steps, self.max_steps + 1, (B,), device=device)
        self.step = 0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        buf_len = self.max_steps + 1
        B = data.shape[0]

        # Write current observation
        self.buffer[self.step % buf_len] = data

        # Read delayed observation (per-env)
        read_idx = (self.step - self.delays) % buf_len
        delayed = self.buffer[read_idx, torch.arange(B, device=data.device)]

        self.step += 1
        return delayed

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            # Full reset
            if self.min_steps == self.max_steps:
                self.delays[:] = self.min_steps
            else:
                self.delays = torch.randint(
                    self.min_steps, self.max_steps + 1,
                    (self.delays.shape[0],), device=self.delays.device,
                )
            self.buffer[:] = 0.0
            self.step = 0
        else:
            env_ids_t = torch.tensor(env_ids, device=self.delays.device, dtype=torch.long) \
                if not isinstance(env_ids, torch.Tensor) else env_ids
            if self.min_steps == self.max_steps:
                self.delays[env_ids_t] = self.min_steps
            else:
                self.delays[env_ids_t] = torch.randint(
                    self.min_steps, self.max_steps + 1,
                    (len(env_ids_t),), device=self.delays.device,
                )
            # Don't zero buffer — let old data be naturally overwritten
            # (zeroing causes false observations that crash training)
