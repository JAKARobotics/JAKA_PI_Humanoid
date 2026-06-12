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

from jaka_rl_lab.assets.jaka import Khan_mini_CFG,Khan_mini_JOINT_NAMES_DEPLOY,Khan_mini_END_LINK_NAMES
from jaka_rl_lab.tasks.locomotion import mdp

JOINT_DATA_ASSET_CFG=SceneEntityCfg(name="robot",joint_names=Khan_mini_JOINT_NAMES_DEPLOY,preserve_order=True)

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
    robot: ArticulationCfg = Khan_mini_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

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

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1),heading=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointActionMixedCfg(
        asset_name="robot", joint_names=Khan_mini_JOINT_NAMES_DEPLOY, scale=0.5)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.waist_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.waist_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5),params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_ang_vel = ObsTerm(func=mdp.waist_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.waist_projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05,params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        last_action = ObsTerm(func=mdp.last_action)

        base_lin_vel = ObsTerm(func=mdp.waist_lin_vel)
        feet_state = ObsTerm(func=mdp.feet_contact,params={"sensor_cfg": SceneEntityCfg(name="contact_forces",body_names=[".*ankle_roll.*"],preserve_order=True)})

        #to do: add feet contact

        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 10

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # alive = RewTerm(func=mdp.is_alive, weight=0.15)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- base
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)   # 1.0
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-4)  # 1.5

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.06,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )

    joint_deviation_elbow = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
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

    # joint_deviation_shoulder = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch_joint",
    #             ],
    #         )
    #     },
    # )

    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                    "Neck_.*",
                ],
            )
        },
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.6,   
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    joint_deviation_motion = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,   #0.08
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*ankle.*"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)#should rely on command
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.58}) 

    # -- feet
    # gait = RewTerm(
    #     func=mdp.feet_gait,
    #     weight=0.5,
    #     params={
    #         "period": 0.8,
    #         "offset": [0.0, 0.5],
    #         "threshold": 0.55,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    #     },
    # )

    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )

    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped,
                            weight=0.2,
                            params={"command_name": "base_velocity",
                                    "sensor_cfg": SceneEntityCfg(name="contact_forces",body_names=[".*ankle_roll.*"]),
                                    "threshold": 0.4})
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll.*"])},
    )

    # feet_clearance = RewTerm(
    #     func=mdp.foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "std": 0.05,
    #         "tanh_mult": 2.0,
    #         "target_height": 0.1,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #     },
    # )

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

    ankle_action=RewTerm(
        func=mdp.penalize_action,
        weight=-1e-3,
        params={
            "joint_ids": [4,5,16,17],
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


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


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

    fbdata_delay=False
    joint_names=Khan_mini_JOINT_NAMES_DEPLOY
    end_link_names=Khan_mini_END_LINK_NAMES

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
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        
        self.commands.base_velocity.ranges=self.commands.base_velocity.limit_ranges


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_type="plane"
        self.scene.terrain.terrain_generator=None
        # self.scene.terrain.terrain_generator.num_rows = 2
        # self.scene.terrain.terrain_generator.num_cols = 5
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.0, 0.0),heading=(0.0,0.0)
        )
        self.events.push_robot=None

import torch 
from rsl_rl.env import VecEnv
def mini_data_augmentation_callback(obs:torch.Tensor|None,actions:torch.Tensor|None,env:VecEnv,obs_type:str="policy"):
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

