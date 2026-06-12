# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Conveniences to other module directories via relative paths
from jaka_rl_lab.assets import ISAAC_ASSET_DIR

# Khan_mini configuration
Khan_mini_JOINT_NAMES_DEPLOY=[
    "Left_hip_pitch_joint", #left leg (6 dof)
    "Left_hip_roll_joint",
    "Left_hip_yaw_joint",
    "Left_knee_joint",
    "Left_ankle_pitch_joint",
    "Left_ankle_roll_joint",
    "Left_shoulder_pitch_joint", #left arm (6 dof)
    "Left_shoulder_roll_joint",
    "Left_shoulder_yaw_joint",
    "Left_elbow_joint",
    "Left_wrist_roll_joint",
    "Left_wrist_yaw_joint",
    "Right_hip_pitch_joint", #right leg (6dof)
    "Right_hip_roll_joint",
    "Right_hip_yaw_joint",
    "Right_knee_joint",
    "Right_ankle_pitch_joint",
    "Right_ankle_roll_joint",
    "Right_shoulder_pitch_joint", #right arm (6 dof)
    "Right_shoulder_roll_joint",
    "Right_shoulder_yaw_joint",
    "Right_elbow_joint",
    "Right_wrist_roll_joint",
    "Right_wrist_yaw_joint",
    "waist_yaw_joint", #waist (1 dof)
    "Neck_yaw_joint",  #neck (2 dof)
    "Neck_pitch_joint",
]

Khan_mini_END_LINK_NAMES=[
    "Left_wrist_yaw_Link",
    "Right_wrist_yaw__Link",
    "Left_ankle_roll_Link",
    "Right_ankle_roll_Link",
]

Khan_mini_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/jaka/Khan_mini_simplified/Khan_mini_simplified.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65), #0.95
        joint_pos={
            "Left_hip_pitch_joint": 0.0,
            "Left_hip_roll_joint": 0.0,
            "Left_hip_yaw_joint": 0.0,
            "Left_knee_joint": 0.0,
            "Left_ankle_pitch_joint": 0.0,
            "Left_ankle_roll_joint": 0.0,
            "Left_shoulder_pitch_joint": 0.0,
            "Left_shoulder_roll_joint": -1.57,
            "Left_shoulder_yaw_joint": 0.0,
            "Left_elbow_joint": 1.57,
            "Left_wrist_roll_joint": 0.0,
            "Left_wrist_yaw_joint": 0.3,

            "Right_hip_pitch_joint": 0.0,
            "Right_hip_roll_joint": 0.0,
            "Right_hip_yaw_joint": 0.0,
            "Right_knee_joint": 0.0, 
            "Right_ankle_pitch_joint": 0.0,
            "Right_ankle_roll_joint": 0.0,
            "Right_shoulder_pitch_joint": 0.0,
            "Right_shoulder_roll_joint": -1.57,
            "Right_shoulder_yaw_joint": 0.0,
            "Right_elbow_joint": 1.57,
            "Right_wrist_roll_joint": 0.0,
            "Right_wrist_yaw_joint": 0.3,

            "waist_yaw_joint": 0.0,
            "Neck_yaw_joint": 0.0,
            "Neck_pitch_joint": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[ #0.62 21.91
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_yaw_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 11.52,
                ".*_hip_roll_joint": 11.52,
                ".*_hip_yaw_joint": 11.52,
                ".*_knee_joint": 11.52,
            },
            stiffness={
                 ".*_hip_pitch_joint": 187.0,
                ".*_hip_roll_joint": 187.0,
                ".*_hip_yaw_joint": 187.0,
                ".*_knee_joint": 187.0,
            },
            damping={
                ".*_hip_pitch_joint": 18.7,
                ".*_hip_roll_joint": 18.7, 
                ".*_hip_yaw_joint": 18.7,
                ".*_knee_joint": 18.7,
            },
            armature=0.03, #JX9-120 model
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=96.0,
            velocity_limit_sim=14.6,          
            stiffness={
                ".*_ankle_pitch_joint": 100.0,
                ".*_ankle_roll_joint": 50.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 0.5,
            },
            armature=0.016,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 96.0,
                ".*_shoulder_roll_joint": 96.0,
                ".*_shoulder_yaw_joint": 36.0,
                ".*_elbow_joint": 36.0,
                ".*_wrist_.*_joint": 8.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 11.5,
                ".*_shoulder_roll_joint": 11.5,
                ".*_shoulder_yaw_joint": 15.7,
                ".*_elbow_joint": 15.7,
                ".*_wrist_.*_joint": 12.5,
            },          
            stiffness={
                ".*_shoulder_pitch_joint": 102.0,
                ".*_shoulder_roll_joint": 102.0,
                ".*_shoulder_yaw_joint": 40.8,
                ".*_elbow_joint": 40.8,
                ".*_wrist_.*_joint": 6.7,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.2,
                ".*_shoulder_roll_joint": 10.2,
                ".*_shoulder_yaw_joint": 4.0,
                ".*_elbow_joint": 4.0,
                ".*_wrist_.*_joint": 0.67,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.016,
                ".*_shoulder_roll_joint": 0.016,
                ".*_shoulder_yaw_joint": 0.01, #0.003
                ".*_elbow_joint": 0.01,
                ".*_wrist_.*_joint": 0.01, #0.0005,
            },
        ),
        "others": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "Neck_yaw_joint", "Neck_pitch_joint"],
            effort_limit_sim={
                "waist_yaw_joint": 120.0,
                "Neck_yaw_joint": 8.0,
                "Neck_pitch_joint": 8.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 11.52,
                "Neck_yaw_joint": 12.5,
                "Neck_pitch_joint": 12.5,
            },          
            stiffness={
                "waist_yaw_joint": 187.0,
                "Neck_yaw_joint": 6.7,
                "Neck_pitch_joint": 6.7,
            },
            damping={
                "waist_yaw_joint": 18.7,
                "Neck_yaw_joint": 0.67,
                "Neck_pitch_joint": 0.67,
            },
            armature={
                "waist_yaw_joint": 0.03,
                "Neck_yaw_joint": 0.01,
                "Neck_pitch_joint": 0.01,
            },
        ),
    },
)