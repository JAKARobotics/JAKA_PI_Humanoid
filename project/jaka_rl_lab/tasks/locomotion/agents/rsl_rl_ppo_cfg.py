# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
# from jaka_rl_lab.tasks.locomotion.mdp import MyRLEnv
from jaka_rl_lab.tasks.locomotion.envs.Khan_mini_27dof import ISAAC_K1L_MINI_LOCO_DIR
from jaka_rl_lab.tasks.locomotion.envs.Khan_mini_27dof.velocity_env_cfg import mini_data_augmentation_callback
# import torch
import glob


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    clip_actions = 10.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    class_name="OnPolicyRunner"


@configclass
class MiniSymPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.algorithm.symmetry_cfg=RslRlSymmetryCfg()
        self.algorithm.symmetry_cfg.use_data_augmentation=True
        self.algorithm.symmetry_cfg.use_mirror_loss=True
        self.algorithm.symmetry_cfg.mirror_loss_coeff=0.5  # 1.0
        self.algorithm.symmetry_cfg.data_augmentation_func=mini_data_augmentation_callback

@configclass
class MiniSymAmpPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        self.class_name="AmpOnPolicyRunner"
        self.algorithm.class_name="AMPPPO"

        self.algorithm.symmetry_cfg=RslRlSymmetryCfg()
        self.algorithm.symmetry_cfg.use_data_augmentation=True
        self.algorithm.symmetry_cfg.use_mirror_loss=True
        self.algorithm.symmetry_cfg.mirror_loss_coeff=2.0
        self.algorithm.symmetry_cfg.data_augmentation_func=mini_data_augmentation_callback

        self.amp_reward_coef=0.15
        self.amp_motion_files=[f"{ISAAC_K1L_MINI_LOCO_DIR}/data.txt"]
        # self.amp_motion_files = glob.glob(f"{ISAAC_K1L_MINI_LOCO_DIR}/amp_data/*.txt")
        self.amp_num_preload_transitions=200000
        self.amp_task_reward_lerp=0.9
        self.amp_discr_hidden_dims=[1024,512,256]
        self.min_normalized_std=[0.05]*20