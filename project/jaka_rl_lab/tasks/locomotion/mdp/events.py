from isaaclab.envs import ManagerBasedEnv
import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.actuators import ImplicitActuator
import isaaclab.utils.math as math_utils


def randomize_pd_gains_same_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distribution: str = "log_uniform",
):
    """Scale stiffness and damping by the same random factor per environment and joint."""
    if scale_range[0] <= 0.0 or scale_range[1] < scale_range[0]:
        raise ValueError(f"Invalid positive PD gain scale range: {scale_range}.")
    if distribution == "uniform":
        sample_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        sample_fn = math_utils.sample_log_uniform
    else:
        raise ValueError(f"Unsupported PD gain scale distribution: '{distribution}'.")

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.as_tensor(actuator.joint_indices, device=asset.device)
        elif isinstance(actuator.joint_indices, slice):
            global_indices = actuator_indices = torch.as_tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            actuator_joint_indices = torch.as_tensor(actuator.joint_indices, device=asset.device)
            selected_joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device)
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, selected_joint_ids)).flatten()
            if len(actuator_indices) == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]

        default_stiffness = asset.data.default_joint_stiffness[env_ids][:, global_indices]
        default_damping = asset.data.default_joint_damping[env_ids][:, global_indices]
        gain_scale = sample_fn(
            scale_range[0],
            scale_range[1],
            default_stiffness.shape,
            device=asset.device,
        )

        stiffness = actuator.stiffness[env_ids].clone()
        damping = actuator.damping[env_ids].clone()
        stiffness[:, actuator_indices] = default_stiffness * gain_scale
        damping[:, actuator_indices] = default_damping * gain_scale
        actuator.stiffness[env_ids] = stiffness
        actuator.damping[env_ids] = damping

        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
            asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)



def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(env.joint_reset_lower_lmt,env.joint_reset_upper_lmt, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids][:, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )
