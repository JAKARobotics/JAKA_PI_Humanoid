from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply_inverse
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.contact_sensor import ContactSensor


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg=SceneEntityCfg("contact_forces"), threshold:float=0.5)->torch.Tensor:
    asset: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces=asset.data.net_forces_w_history
    feet_state=torch.max(torch.norm(net_contact_forces[:, :,sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return feet_state

def delay_root_quat_w(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.root_quat_buffer is not None:
        return env.root_quat_buffer._circular_buffer[ env.root_quat_buffer._time_lags].clone()
    else:
        asset: RigidObject = env.scene[asset_cfg.name]
        return asset.data.root_quat_w

def delay_root_omega_b(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.root_omega_buffer is not None:
        return env.root_omega_buffer._circular_buffer[ env.root_omega_buffer._time_lags].clone()
    else:
        asset: RigidObject = env.scene[asset_cfg.name]
        return asset.data.root_ang_vel_b

def delay_joint_pos(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.joint_pos_rel_buffer is not None:
        return env.joint_pos_buffer._circular_buffer[env.joint_pos_rel_buffer._time_lags].clone()
    else:
        asset: Articulation = env.scene[asset_cfg.name]
        return asset.data.joint_pos[:, asset_cfg.joint_ids]

def delay_joint_vel(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if env.joint_vel_rel_buffer is not None:
        return env.joint_vel_buffer._circular_buffer[env.joint_vel_rel_buffer._time_lags].clone()
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

def waist_projected_gravity(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "waist_id"):
        body_ids, _ = asset.find_bodies("waist_yaw_Link")
        env.waist_id = body_ids[0]

    quat_w = asset.data.body_quat_w[:, env.waist_id, :]
    gravity_w = quat_w.new_tensor([0.0, 0.0, -1.0]).repeat(quat_w.shape[0], 1)
    gravity_b = quat_apply_inverse(quat_w, gravity_w)

    return gravity_b

def waist_lin_vel(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "waist_id"):
        body_ids, _ = asset.find_bodies("waist_yaw_Link")
        env.waist_id = body_ids[0]

    lin_vel_w = asset.data.body_lin_vel_w[:, env.waist_id, :]
    quat_w = asset.data.body_quat_w[:, env.waist_id, :]
    lin_vel_b = quat_apply_inverse(quat_w, lin_vel_w)

    return lin_vel_b