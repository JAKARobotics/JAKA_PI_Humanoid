from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class UniformVelCmd(UniformVelocityCommand):
    """Sample velocity commands uniformly from inside an ellipsoid.

    The configured ranges define the ellipsoid's axis-aligned bounding box in
    ``(lin_vel_x, lin_vel_y, ang_vel_z)`` space.
    """

    cfg: UniformVelCmdCfg

    def __init__(self, cfg: UniformVelCmdCfg, env):
        super().__init__(cfg, env)

        if self.cfg.max_lin_accel <= 0.0:
            raise ValueError("max_lin_accel must be positive.")
        if self.cfg.max_ang_accel <= 0.0:
            raise ValueError("max_ang_accel must be positive.")

        # Resampling changes this target only. ``vel_command_b`` remains the
        # acceleration-limited command exposed through the command property.
        self._target_vel_command_b = torch.zeros_like(self.vel_command_b)

    @property
    def command(self) -> torch.Tensor:
        """The acceleration-limited velocity command exposed to the environment."""
        return self.vel_command_b

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands uniformly by volume inside the configured ellipsoid."""
        num_envs = len(env_ids)
        if num_envs == 0:
            return

        bounds = torch.tensor(
            (
                self.cfg.ranges.lin_vel_x,
                self.cfg.ranges.lin_vel_y,
                self.cfg.ranges.ang_vel_z,
            ),
            device=self.device,
            dtype=self.vel_command_b.dtype,
        )
        center = bounds.mean(dim=1)
        semi_axes = 0.5 * (bounds[:, 1] - bounds[:, 0])

        # A normalized Gaussian is uniform on the unit sphere. Scaling its
        # radius by U**(1/3) makes samples uniform throughout the ball's volume.
        direction = torch.randn((num_envs, 3), device=self.device, dtype=self.vel_command_b.dtype)
        direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(
            torch.finfo(direction.dtype).tiny
        )
        radius = torch.rand((num_envs, 1), device=self.device, dtype=self.vel_command_b.dtype).pow_(1.0 / 3.0)
        self._target_vel_command_b[env_ids] = center + semi_axes * direction * radius

        random_values = torch.empty(num_envs, device=self.device)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = random_values.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = random_values.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        self.is_standing_env[env_ids] = random_values.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Update the target command, then approach it subject to acceleration limits."""
        if self.cfg.heading_command:
            heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[heading_env_ids] - self.robot.data.heading_w[heading_env_ids]
            )
            self._target_vel_command_b[heading_env_ids, 2] = torch.clamp(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self._target_vel_command_b[standing_env_ids] = 0.0

        # Limit the norm of the xy velocity change so diagonal commands obey
        # the same linear-acceleration bound as axis-aligned commands.
        delta_lin_vel = self._target_vel_command_b[:, :2] - self.vel_command_b[:, :2]
        max_lin_vel_step = self.cfg.max_lin_accel * self._env.step_dt
        delta_lin_vel_norm = torch.linalg.vector_norm(delta_lin_vel, dim=1, keepdim=True)
        lin_vel_scale = torch.clamp(
            max_lin_vel_step / delta_lin_vel_norm.clamp_min(torch.finfo(delta_lin_vel.dtype).tiny),
            max=1.0,
        )
        self.vel_command_b[:, :2] += delta_lin_vel * lin_vel_scale

        delta_ang_vel = self._target_vel_command_b[:, 2] - self.vel_command_b[:, 2]
        max_ang_vel_step = self.cfg.max_ang_accel * self._env.step_dt
        self.vel_command_b[:, 2] += torch.clamp(
            delta_ang_vel,
            min=-max_ang_vel_step,
            max=max_ang_vel_step,
        )


@configclass
class UniformVelCmdCfg(UniformVelocityCommandCfg):
    """Configuration for ellipsoid-uniform velocity commands."""

    class_type: type = UniformVelCmd

    max_lin_accel: float = 1.0
    """Maximum xy command acceleration magnitude in m/s^2."""

    max_ang_accel: float = 2.0
    """Maximum yaw command acceleration magnitude in rad/s^2."""

    # limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
