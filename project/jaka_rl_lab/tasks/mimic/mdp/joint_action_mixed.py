from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class JointActionMixed(ActionTerm):
    cfg: JointActionMixedCfg
    _asset: Articulation
    _scale: torch.Tensor | float
    _offset: torch.Tensor | float
    _clip: torch.Tensor

    def __init__(self, cfg: JointActionMixedCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        
        # parse offset, we use defaul_joint_pos here
        self._offset = self._asset.data.default_joint_pos.clone()

        # parse clip
        self._clip = self._asset.data.default_joint_pos_limits[:, self._joint_ids].clone()

        # ``process_actions`` is called at the environment rate, while
        # ``apply_actions`` is called once per simulation step.  Keep the
        # previously applied target so the new target can be reached linearly
        # over all env_dt / sim_dt simulation sub-steps.
        self._interpolation_steps = int(round(env.step_dt / env.physics_dt))
        if self._interpolation_steps < 1:
            raise ValueError(
                "The environment step time must be greater than or equal to the physics step time. "
                f"Received env_dt={env.step_dt} and sim_dt={env.physics_dt}."
            )
        self._interpolation_step = self._interpolation_steps
        self._interpolation_start = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._interpolated_actions = self._interpolation_start.clone()
        self._applied_actions = self._interpolation_start.clone()

        # Resolve the random action delay. A maximum of ``None`` means one
        # complete environment step. The extra history slot stores the current
        # command, allowing delays from zero through the configured maximum.
        min_delay, max_delay = self.cfg.action_delay_range
        max_delay = self._interpolation_steps if max_delay is None else max_delay
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError(
                "action_delay_range must satisfy 0 <= min_delay <= max_delay. "
                f"Received {self.cfg.action_delay_range}."
            )
        if max_delay > self._interpolation_steps:
            raise ValueError(
                "The maximum action delay cannot exceed env_dt / sim_dt. "
                f"Received max_delay={max_delay}, but env_dt / sim_dt={self._interpolation_steps}."
            )
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_history = self._interpolation_start.unsqueeze(0).repeat(max_delay + 1, 1, 1)
        self._history_write_index = 0
        self._env_indices = torch.arange(self.num_envs, device=self.device)


    """
    Properties.
    """
    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # Continue the undelayed command trajectory from the preceding
        # simulation sub-step.
        self._interpolation_start[:] = self._interpolated_actions
        self._interpolation_step = 0
        self._action_delay_steps.random_(self._min_delay, self._max_delay + 1)
        # apply the affine transformations
        self._processed_actions[:] = self._raw_actions * self._scale + self._offset[:, self._joint_ids]
        # clip actions
        # self._processed_actions = torch.clamp(
        #     self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
        # )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        current_joint_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._interpolation_start[env_ids] = current_joint_pos
        self._interpolated_actions[env_ids] = current_joint_pos
        self._applied_actions[env_ids] = current_joint_pos
        self._action_delay_steps[env_ids] = 0
        self._action_history[:, env_ids] = current_joint_pos.unsqueeze(0)

    def apply_actions(self):
        # Reach the processed action exactly on the final simulation sub-step.
        if self._interpolation_step < self._interpolation_steps:
            self._interpolation_step += 1
            alpha = self._interpolation_step / self._interpolation_steps
            torch.lerp(
                self._interpolation_start,
                self._processed_actions,
                alpha,
                out=self._interpolated_actions,
            )
        else:
            self._interpolated_actions[:] = self._processed_actions

        # Cache the current undelayed command, then select a different history
        # slot for each environment according to its sampled sub-step delay.
        self._action_history[self._history_write_index] = self._interpolated_actions
        read_indices = (self._history_write_index - self._action_delay_steps) % len(self._action_history)
        self._applied_actions[:] = self._action_history[read_indices, self._env_indices]
        self._history_write_index = (self._history_write_index + 1) % len(self._action_history)

        self._asset.set_joint_position_target(self._applied_actions, joint_ids=self._joint_ids)

    
@configclass
class JointActionMixedCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    preserve_order: bool = True
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""
    use_default_offset: bool = True
    action_delay_range: tuple[int, int | None] = (0, 4)
    """Inclusive random action delay range in simulation steps.

    The delay is sampled independently for every environment at each control
    step. A maximum of ``None`` uses ``env_dt / sim_dt``. Set this to ``(0, 0)``
    to disable action delay.
    """

    class_type: type[ActionTerm] = JointActionMixed
