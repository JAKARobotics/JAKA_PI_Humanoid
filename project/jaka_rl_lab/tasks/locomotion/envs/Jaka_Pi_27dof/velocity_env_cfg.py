import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from jaka_rl_lab.assets.jaka import JAKA_PI_CFG,JAKA_PI_JOINT_NAMES_DEPLOY,JAKA_PI_END_LINK_NAMES
from jaka_rl_lab.tasks.locomotion import mdp

JOINT_DATA_ASSET_CFG=SceneEntityCfg(name="robot",joint_names=JAKA_PI_JOINT_NAMES_DEPLOY,preserve_order=True)

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)

GRAVEL_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=GRAVEL_TERRAINS_CFG, #COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        # max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = JAKA_PI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    ankle_roll_pair_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Left_ankle_roll_Link",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Right_ankle_roll_Link"],
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_Link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    add_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot",body_names="waist_yaw_Link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    joint_pd_gains = EventTerm(
        func=mdp.randomize_pd_gains_same_scale,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "scale_range": (0.8, 1.2),
            "distribution": "log_uniform",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelCmdCfg( 
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        max_lin_accel=5.0,
        max_ang_accel=10.0,
        debug_vis=True,
        ranges=mdp.UniformVelCmdCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointActionMixedCfg(
        asset_name="robot", 
        joint_names=JAKA_PI_JOINT_NAMES_DEPLOY, scale=0.5,
        action_delay_range=(0,1))


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.delayed_root_omega_b, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.delayed_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5),params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05,params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        last_action = ObsTerm(func=mdp.last_action)

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        feet_state = ObsTerm(func=mdp.feet_contact,params={"sensor_cfg": SceneEntityCfg(name="contact_forces",body_names=[".*ankle_roll.*"],preserve_order=True)})
        feet_force = ObsTerm(func=mdp.feet_force, scale=0.001, params={"sensor_cfg": SceneEntityCfg(name="contact_forces",body_names=[".*ankle_roll.*"],preserve_order=True)})
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=1.0/120, params={"asset_cfg": JOINT_DATA_ASSET_CFG})

        def __post_init__(self):
            self.history_length = 10

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        # func=mdp.track_lin_vel_xy_yaw_frame_exp,
        func=mdp.track_lin_vel_xy_exp_adaptive_std,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_adaptive_std, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # alive = RewTerm(func=mdp.is_alive, weight=0.15)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- base
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)   # 1.0
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    joint_velocity_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9},
    )
    joint_effort_limits = RewTerm(
        func=mdp.joint_effort_limits,
        weight=-0.01,
        params={"soft_factor": 0.9},
    )
    energy = RewTerm(func=mdp.energy, weight=-1.0e-4)  # 1.5


    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_elbow_joint",
                ],
            )
        },
    )

    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                ],
            )
        },
    )


    joint_deviation_head = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Neck_.*",
                ],
            )
        },
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,   
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)#should rely on command
    # base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.58}) 


    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "asset_cfg":SceneEntityCfg("robot",body_names="waist_yaw_Link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 500,#root link mass is not included
            "max_reward": 400,
        },
    )


    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle_roll.*).*"]),
        },
    )

    undesired_ankle_roll_pair_contact = RewTerm(
        func=mdp.undesired_contact_pair,
        weight=-0.5,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("ankle_roll_pair_contact"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    excessive_lin_vel_xy_error = DoneTerm(
        func=mdp.excessive_lin_vel_xy_tracking_error,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "duration": 3.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""



@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    imu_delay=True
    omega_max_delay=4
    quat_max_extra_delay=4
    joint_names=JAKA_PI_JOINT_NAMES_DEPLOY
    end_link_names=JAKA_PI_END_LINK_NAMES

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.ankle_roll_pair_contact.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_type="plane"
        self.scene.terrain.terrain_generator=None
        # self.scene.terrain.terrain_generator.num_rows = 2
        # self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0),heading=(0.0,0.0)
        )
        self.events.push_robot=None

import torch 
from rsl_rl.env import VecEnv
def sym_augmentation_callback(obs:torch.Tensor|None,actions:torch.Tensor|None,env:VecEnv,obs_type:str="policy"):
    if obs==None:
        return_obs=None
    else:
        if obs_type=="policy":
            obs_history_len=env.unwrapped.cfg.observations.policy.history_length
        elif obs_type=="critic":
            obs_history_len=env.unwrapped.cfg.observations.critic.history_length
        else: 
            obs_history_len=1

        single_obs_num =obs.shape[1]//obs_history_len
        mirrored_obs=obs.clone()
        for i in range(obs_history_len):
            mirrored_obs[:,single_obs_num*i+0]=-obs[:,single_obs_num*i+0]#root_angular_vel
            mirrored_obs[:,single_obs_num*i+1]=obs[:,single_obs_num*i+1]
            mirrored_obs[:,single_obs_num*i+2]=-obs[:,single_obs_num*i+2]
            mirrored_obs[:,single_obs_num*i+3]=obs[:,single_obs_num*i+3]#projected_gravity
            mirrored_obs[:,single_obs_num*i+4]=-obs[:,single_obs_num*i+4]
            mirrored_obs[:,single_obs_num*i+5]=obs[:,single_obs_num*i+5]
            mirrored_obs[:,single_obs_num*i+6]=obs[:,single_obs_num*i+6]#command
            mirrored_obs[:,single_obs_num*i+7]=-obs[:,single_obs_num*i+7]
            mirrored_obs[:,single_obs_num*i+8]=-obs[:,single_obs_num*i+8]
            mirrored_obs[:,single_obs_num*i+9:single_obs_num*i+21]=obs[:,single_obs_num*i+21:single_obs_num*i+33]#joint_pos
            mirrored_obs[:,single_obs_num*i+21:single_obs_num*i+33]=obs[:,single_obs_num*i+9:single_obs_num*i+21]
            mirrored_obs[:,single_obs_num*i+33:single_obs_num*i+35]=-obs[:,single_obs_num*i+33:single_obs_num*i+35]#waist and neck_yaw
            mirrored_obs[:,single_obs_num*i+36:single_obs_num*i+48]=obs[:,single_obs_num*i+48:single_obs_num*i+60]#joint_vel
            mirrored_obs[:,single_obs_num*i+48:single_obs_num*i+60]=obs[:,single_obs_num*i+36:single_obs_num*i+48]
            mirrored_obs[:,single_obs_num*i+60:single_obs_num*i+62]=-obs[:,single_obs_num*i+60:single_obs_num*i+62]#waist and neck_yaw
            mirrored_obs[:,single_obs_num*i+63:single_obs_num*i+75]=obs[:,single_obs_num*i+75:single_obs_num*i+87]#last action
            mirrored_obs[:,single_obs_num*i+75:single_obs_num*i+87]=obs[:,single_obs_num*i+63:single_obs_num*i+75]
            mirrored_obs[:,single_obs_num*i+87:single_obs_num*i+89]=-obs[:,single_obs_num*i+87:single_obs_num*i+89]#waist and neck_yaw

            if obs_type=="critic":
                mirrored_obs[:,single_obs_num*i+90]=obs[:,single_obs_num*i+90]#root_lin_vel
                mirrored_obs[:,single_obs_num*i+91]=-obs[:,single_obs_num*i+91]
                mirrored_obs[:,single_obs_num*i+92]=obs[:,single_obs_num*i+92]
                mirrored_obs[:,single_obs_num*i+93]=obs[:,single_obs_num*i+94]#feet_contact
                mirrored_obs[:,single_obs_num*i+94]=obs[:,single_obs_num*i+93]
                mirrored_obs[:,single_obs_num*i+95]=obs[:,single_obs_num*i+96]#feet_force
                mirrored_obs[:,single_obs_num*i+96]=obs[:,single_obs_num*i+95]
                mirrored_obs[:,single_obs_num*i+97:single_obs_num*i+109]=obs[:,single_obs_num*i+109:single_obs_num*i+121]#joint_effort
                mirrored_obs[:,single_obs_num*i+109:single_obs_num*i+121]=obs[:,single_obs_num*i+97:single_obs_num*i+109]
                mirrored_obs[:,single_obs_num*i+121:single_obs_num*i+123]=-obs[:,single_obs_num*i+121:single_obs_num*i+123]#waist and neck_yaw effort

        return_obs=torch.vstack((obs,mirrored_obs))


    if actions==None:
        return_action=None
    else:
        mirrored_action=actions.clone()
        mirrored_action[:,:12]=actions[:,12:24]#right 
        mirrored_action[:,12:24]=actions[:,:12]#left
        mirrored_action[:,24:26]=-actions[:,24:26]#waist and neck_yaw
        return_action=torch.vstack((actions,mirrored_action))
        

    return return_obs,return_action
