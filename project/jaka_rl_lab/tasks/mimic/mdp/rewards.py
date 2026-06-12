from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from jaka_rl_lab.tasks.mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_pos_error_exp(env: ManagerBasedRLEnv, command_name: str,std:float)->torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error=torch.sum(
        torch.square(command.joint_pos - command.robot_joint_pos), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_joint_vel_error_exp(env: ManagerBasedRLEnv, command_name: str,std:float)->torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error=torch.sum(
        torch.square(command.joint_vel - command.robot_joint_vel), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def swing_foot_touch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    foot_body_names: list[str],
    contact_threshold: float = 8.0,
    height_threshold: float = 0.035,
    horizontal_speed_threshold: float = 0.18,
    xy_speed_only_contact_condition: bool = False,
) -> torch.Tensor:
    """Penalize foot contact when the reference motion indicates the foot should still be swinging."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_indexes = _get_body_indexes(command, foot_body_names)
    if len(foot_indexes) != len(foot_body_names):
        raise ValueError(f"Could not find all foot bodies in motion command: {foot_body_names}")

    target_foot_height = command.body_pos_w[:, foot_indexes, 2]
    target_foot_horizontal_speed = torch.norm(command.body_lin_vel_w[:, foot_indexes, :2], dim=-1)

    if xy_speed_only_contact_condition:
        desired_contact = target_foot_horizontal_speed < horizontal_speed_threshold
    else:
        desired_contact = (target_foot_height < height_threshold) & (target_foot_horizontal_speed < horizontal_speed_threshold)
    double_support = torch.all(desired_contact, dim=1, keepdim=True)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    actual_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    )

    unwanted_contact = actual_contact & ~desired_contact & ~double_support
    return torch.sum(unwanted_contact.float(), dim=1)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
