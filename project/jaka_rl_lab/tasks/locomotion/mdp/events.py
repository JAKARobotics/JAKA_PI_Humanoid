from isaaclab.envs import ManagerBasedEnv
import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.assets import Articulation, DeformableObject, RigidObject
import isaaclab.utils.math as math_utils



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
