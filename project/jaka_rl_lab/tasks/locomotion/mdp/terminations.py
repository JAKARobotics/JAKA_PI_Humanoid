from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


class excessive_lin_vel_xy_tracking_error(ManagerTermBase):
    """Terminate after the yaw-frame xy velocity error stays excessive for a given duration."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        threshold = cfg.params["threshold"]
        duration = cfg.params["duration"]
        if threshold < 0.0:
            raise ValueError("Linear velocity tracking-error threshold must be non-negative.")
        if duration <= 0.0:
            raise ValueError("Linear velocity tracking-error duration must be positive.")

        self._required_steps = max(1, math.ceil(duration / env.step_dt))
        self._consecutive_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Clear the consecutive-error counter for reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float,
        duration: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        base_lin_vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
        command_xy = env.command_manager.get_command(command_name)[:, :2]
        error_sq = torch.sum(torch.square(command_xy - base_lin_vel_yaw[:, :2]), dim=1)
        above_threshold = error_sq > threshold**2

        self._consecutive_steps = torch.where(
            above_threshold,
            self._consecutive_steps + 1,
            torch.zeros_like(self._consecutive_steps),
        )
        return self._consecutive_steps >= self._required_steps
