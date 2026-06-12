from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
# from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from jaka_rl_lab.assets.jaka import Khan_mini_CFG, Khan_mini_JOINT_NAMES_DEPLOY

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import jaka_rl_lab.tasks.mimic.mdp as mdp


JOINT_DATA_ASSET_CFG=SceneEntityCfg(name="robot",joint_names=Khan_mini_JOINT_NAMES_DEPLOY,preserve_order=True)

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.6, 0.6),
}

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    # curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        # "high_platform": terrain_gen.MeshPitTerrainCfg(
        #     proportion=0.02, pit_depth_range=(0.0, 0.2), platform_width=2.0, double_pit=True
        # ),
        # "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.04, amplitude_range=(0.0, 0.2), num_waves=5.0),
    },
)

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = Khan_mini_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=f"{os.path.dirname(__file__)}/moxingwu_edit.npz",
        anchor_body_name="waist_yaw_Link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=[
            "base_link",
            "Left_hip_yaw_Link",
            "Left_knee_Link",
            "Left_ankle_roll_Link",
            "Right_hip_yaw_Link",
            "Right_knee_Link",
            "Right_ankle_roll_Link",
            "waist_yaw_Link",
            "Left_shoulder_yaw_Link",
            "Left_elbow_Link",
            "Left_wrist_yaw_Link",
            "Right_shoulder_yaw_Link",
            "Right_elbow_Link",
            "Right_wrist_yaw__Link",
            "Neck_pitch_Link"
        ],
        joint_names=Khan_mini_JOINT_NAMES_DEPLOY,
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointActionMixedCfg(asset_name="robot", scale=0.5, joint_names=Khan_mini_JOINT_NAMES_DEPLOY)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.waist_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.delay_joint_pos_rel, noise=Unoise(n_min=-0.02, n_max=0.02), params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        joint_vel_rel = ObsTerm(func=mdp.delay_joint_vel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5), params={"asset_cfg": JOINT_DATA_ASSET_CFG})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.waist_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.waist_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.history_length=5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": JOINT_DATA_ASSET_CFG,
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_Link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot",body_names="waist_yaw_Link"),
            "mass_distribution_params": (-7,7),
            "operation": "add",
        },
    )

    # hand_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot",body_names=["Left_wrist_yaw_Link","Right_wrist_yaw__Link"]),
    #         "mass_distribution_params": (0,5),
    #         "operation": "add",
    #     },
    # )

    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    joint_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip.*",".*knee.*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=3.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=3.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 2.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    # motion_joint_pos = RewTerm(
    #     func=mdp.motion_joint_pos_error_exp,
    #     weight=0.5,
    #     params={"command_name": "motion", "std": 1},
    # )

    # motion_joint_vel = RewTerm(
    #     func=mdp.motion_joint_vel_error_exp,
    #     weight=0.5,
    #     params={"command_name": "motion", "std": 10},
    # )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!Left_ankle_roll_Link$)(?!Right_ankle_roll_Link$)(?!Left_wrist_roll_Link$)(?!Right_wrist_roll_Link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.4},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.4},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.5,
            "body_names": [
                # "Left_ankle_roll_Link",
                # "Right_ankle_roll_Link",
                "Left_wrist_yaw_Link",
                "Right_wrist_yaw__Link",
                "Neck_pitch_Link"
            ],
        },
    )

@configclass
class K1LMiniMimicEnvCfg(ManagerBasedRLEnvCfg):
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
    curriculum = None

    fbdata_delay=True
    joint_names=Khan_mini_JOINT_NAMES_DEPLOY

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class K1LMiniMimicPlayEnvCfg(K1LMiniMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.scene.terrain.terrain_type="plane"
        self.scene.terrain.terrain_generator=None
        self.events.push_robot=None
        