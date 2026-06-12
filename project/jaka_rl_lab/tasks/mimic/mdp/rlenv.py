from __future__ import annotations

# import gymnasium as gym
# import math
# import numpy as np
import torch
# from collections.abc import Sequence
# from typing import Any, ClassVar

# from isaacsim.core.version import get_version

# from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
# from isaaclab.ui.widgets import ManagerLiveVisualizer

from isaaclab.envs.common import VecEnvStepReturn
# from .manager_based_env import ManagerBasedEnv
# from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.buffers import DelayBuffer

class MyRLEnv(ManagerBasedRLEnv):
    joint_ids: list[int] |slice = slice(None)
    root_quat_buffer: DelayBuffer = None
    root_omega_buffer: DelayBuffer = None
    joint_pos_rel_buffer: DelayBuffer = None
    joint_vel_rel_buffer: DelayBuffer = None
    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.joint_names is not None:
            self.joint_ids,_=self.scene["robot"].find_joints(self.cfg.joint_names,preserve_order=True)
        if self.cfg.fbdata_delay:
            # time_lags = torch.randint(
            #     low=0,
            #     high=10,
            #     size=(self.num_envs,),
            #     dtype=torch.int,
            #     device=self.device,
            # )
            time_lags = torch.zeros(self.num_envs,dtype=torch.int,device=self.device)
            self.root_quat_buffer=DelayBuffer(10,self.num_envs,device=self.device)
            self.root_quat_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
            self.root_omega_buffer=DelayBuffer(10,self.num_envs,device=self.device)
            self.root_omega_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
            self.joint_pos_rel_buffer=DelayBuffer(10,self.num_envs,device=self.device)
            self.joint_pos_rel_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
            self.joint_vel_rel_buffer=DelayBuffer(10,self.num_envs,device=self.device)
            self.joint_vel_rel_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
            for _ in range(10):
                self.root_quat_buffer.compute(self.scene["robot"].data.root_quat_w.clone())
                self.root_omega_buffer.compute(self.scene["robot"].data.root_ang_vel_b.clone())
                joint_pos=self.scene["robot"].data.joint_pos[:,self.joint_ids]-self.scene["robot"].data.default_joint_pos[:,self.joint_ids]
                self.joint_pos_rel_buffer.compute(joint_pos.clone())
                self.joint_vel_rel_buffer.compute(self.scene["robot"].data.joint_vel[:,self.joint_ids].clone())
            




    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            if self.root_quat_buffer is not None:
                self.root_quat_buffer.compute(self.scene["robot"].data.root_quat_w.clone())
                self.root_omega_buffer.compute(self.scene["robot"].data.root_ang_vel_b.clone())
                joint_pos=self.scene["robot"].data.joint_pos[:,self.joint_ids]-self.scene["robot"].data.default_joint_pos[:,self.joint_ids]
                self.joint_pos_rel_buffer.compute(joint_pos.clone())
                self.joint_vel_rel_buffer.compute(self.scene["robot"].data.joint_vel[:,self.joint_ids].clone())

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            if self.root_quat_buffer is not None:
                self.root_quat_buffer.reset(reset_env_ids)
                self.root_omega_buffer.reset(reset_env_ids)
                self.joint_pos_rel_buffer.reset(reset_env_ids)
                self.joint_vel_rel_buffer.reset(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.root_quat_buffer is not None:
                for _ in range(10):
                    self.root_quat_buffer.compute(self.scene["robot"].data.root_quat_w.clone())
                    self.root_omega_buffer.compute(self.scene["robot"].data.root_ang_vel_b.clone())
                    joint_pos=self.scene["robot"].data.joint_pos[:,self.joint_ids]-self.scene["robot"].data.default_joint_pos[:,self.joint_ids]
                    self.joint_pos_rel_buffer.compute(joint_pos.clone())
                    self.joint_vel_rel_buffer.compute(self.scene["robot"].data.joint_vel[:,self.joint_ids].clone())

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras