from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from jaka_rl_lab.tasks.mimic.mdp.commands import MotionCommand

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply_inverse
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs import ManagerBasedRLEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def motion_root_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_anchor_quat_w=delay_root_quat_w(env)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def delay_root_quat_w(env: ManagerBasedEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.root_quat_buffer is not None:
        return env.root_quat_buffer._circular_buffer[ env.root_quat_buffer._time_lags].clone()
    else:
        asset: RigidObject = env.scene[asset_cfg.name]
        return asset.data.root_quat_w

def delay_root_omega_b(env: ManagerBasedEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.root_omega_buffer is not None:
        return env.root_omega_buffer._circular_buffer[ env.root_omega_buffer._time_lags].clone()
    else:
        asset: RigidObject = env.scene[asset_cfg.name]
        return asset.data.root_ang_vel_b

def delay_joint_pos_rel(env: ManagerBasedEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.joint_pos_rel_buffer is not None:
        return env.joint_pos_rel_buffer._circular_buffer[env.joint_pos_rel_buffer._time_lags].clone()
    else:
        asset: Articulation = env.scene[asset_cfg.name]
        return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

def delay_joint_vel(env: ManagerBasedEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.joint_vel_rel_buffer is not None:
        return env.joint_vel_rel_buffer._circular_buffer[env.joint_vel_rel_buffer._time_lags].clone()
    else:
        asset: Articulation = env.scene[asset_cfg.name]
        return asset.data.joint_vel[:, asset_cfg.joint_ids]
    
def waist_ang_vel(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "waist_id"):
        body_ids, _ = asset.find_bodies("waist_yaw_Link")
        env.waist_id = body_ids[0]

    ang_vel_w = asset.data.body_ang_vel_w[:, env.waist_id, :]
    quat_w = asset.data.body_quat_w[:, env.waist_id, :]
    ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)

    return ang_vel_b

def waist_lin_vel(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "waist_id"):
        body_ids, _ = asset.find_bodies("waist_yaw_Link")
        env.waist_id = body_ids[0]

    lin_vel_w = asset.data.body_lin_vel_w[:, env.waist_id, :]
    quat_w = asset.data.body_quat_w[:, env.waist_id, :]
    lin_vel_b = quat_apply_inverse(quat_w, lin_vel_w)

    return lin_vel_b