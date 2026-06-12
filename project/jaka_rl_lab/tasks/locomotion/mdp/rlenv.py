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
from isaaclab.utils.buffers import CircularBuffer,DelayBuffer
from isaaclab.managers.observation_manager import ObservationManager
from isaaclab.utils import noise
from isaaclab.utils.math import quat_apply,quat_conjugate

class MyObservationManager(ObservationManager):
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        # check that cfg is not None
        if cfg is None:
            raise ValueError("Observation manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)

    def compute_group(self, group_name, update_history = False)-> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the observations for a given group.

        The observations for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        The following steps are performed for each observation term:

        1. Compute observation term by calling the function
        2. Apply custom modifiers in the order specified in :attr:`ObservationTermCfg.modifiers`
        3. Apply corruption/noise model based on :attr:`ObservationTermCfg.noise`
        4. Apply clipping based on :attr:`ObservationTermCfg.clip`
        5. Apply scaling based on :attr:`ObservationTermCfg.scale`

        We apply noise to the computed term first to maintain the integrity of how noise affects the data
        as it truly exists in the real world. If the noise is applied after clipping or scaling, the noise
        could be artificially constrained or amplified, which might misrepresent how noise naturally occurs
        in the data.

        Args:
            group_name: The name of the group for which to compute the observations. Defaults to None,
                in which case observations for all the groups are computed and returned.
            update_history: The boolean indicator without return obs should be appended to observation group's history.
                Default to False, in which case calling compute_group does not modify history. This input is no-ops
                if the group's history_length == 0.

        Returns:
            Depending on the group's configuration, the tensors for individual observation terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        # evaluate terms: compute, add noise, clip, scale, custom modifiers
        if (group_name=="policy" and self.cfg.policy.history_length is not None) or \
           (group_name=="critic" and self.cfg.critic.history_length is not None):
            for term_name, term_cfg in obs_terms:
                # compute term's value
                obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
                if term_cfg.modifiers is not None:
                    for modifier in term_cfg.modifiers:
                        obs = modifier.func(obs, **modifier.params)
                if isinstance(term_cfg.noise, noise.NoiseCfg):
                    obs = term_cfg.noise.func(obs, term_cfg.noise)
                elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                    obs = term_cfg.noise.func(obs)
                if term_cfg.clip:
                    obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
                if term_cfg.scale is not None:
                    obs = obs.mul_(term_cfg.scale)
                # Update the history buffer if observation term has history enabled
                if term_cfg.history_length > 0:
                    circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                    if update_history:
                        circular_buffer.append(obs)
                    elif circular_buffer._buffer is None:
                        # because circular buffer only exits after the simulation steps,
                        # this guards history buffer from corruption by external calls before simulation start
                        circular_buffer = CircularBuffer(
                            max_len=circular_buffer.max_length,
                            batch_size=circular_buffer.batch_size,
                            device=circular_buffer.device,
                        )
                        circular_buffer.append(obs)

                    #flatten after group concatenation
                    group_obs[term_name] = circular_buffer.buffer
                else:
                    group_obs[term_name] = obs

            # concatenate all observations in the group together
            if self._group_obs_concatenate[group_name]:
                # set the concatenate dimension, account for the batch dimension if positive dimension is given
                return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]).reshape(self._env.num_envs,-1)
            else:
                return group_obs
        else:
            for term_name, term_cfg in obs_terms:
                # compute term's value
                obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
                # apply post-processing
                if term_cfg.modifiers is not None:
                    for modifier in term_cfg.modifiers:
                        obs = modifier.func(obs, **modifier.params)
                if isinstance(term_cfg.noise, noise.NoiseCfg):
                    obs = term_cfg.noise.func(obs, term_cfg.noise)
                elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
                    obs = term_cfg.noise.func(obs)
                if term_cfg.clip:
                    obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
                if term_cfg.scale is not None:
                    obs = obs.mul_(term_cfg.scale)
                # Update the history buffer if observation term has history enabled
                if term_cfg.history_length > 0:
                    circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                    if update_history:
                        circular_buffer.append(obs)
                    elif circular_buffer._buffer is None:
                        # because circular buffer only exits after the simulation steps,
                        # this guards history buffer from corruption by external calls before simulation start
                        circular_buffer = CircularBuffer(
                            max_len=circular_buffer.max_length,
                            batch_size=circular_buffer.batch_size,
                            device=circular_buffer.device,
                        )
                        circular_buffer.append(obs)

                    if term_cfg.flatten_history_dim:
                        group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
                    else:
                        group_obs[term_name] = circular_buffer.buffer
                else:
                    group_obs[term_name] = obs

            # concatenate all observations in the group together
            if self._group_obs_concatenate[group_name]:
                # set the concatenate dimension, account for the batch dimension if positive dimension is given
                return torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
            else:
                return group_obs

class MyRLEnv(ManagerBasedRLEnv):
    joint_ids: list[int] |slice = slice(None)
    end_link_ids: list[int] | slice=slice(None)
    root_quat_buffer: DelayBuffer = None
    root_omega_buffer: DelayBuffer = None
    joint_pos_buffer: DelayBuffer = None
    joint_vel_buffer: DelayBuffer = None
    hand_offset:torch.Tensor = None
    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.hand_offset=torch.zeros(self.num_envs,3,dtype=torch.float32,device=self.device)
        # self.hand_offset[:,2]=-0.24
        if self.cfg.joint_names is not None:
            self.joint_ids,_=self.scene["robot"].find_joints(self.cfg.joint_names,preserve_order=True)
        if self.cfg.end_link_names is not None:
            self.end_link_ids,_=self.scene["robot"].find_bodies(self.cfg.end_link_names,preserve_order=True)
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
            self.joint_pos_buffer=DelayBuffer(10,self.num_envs,device=self.device)
            self.joint_pos_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
            self.joint_vel_buffer=DelayBuffer(10,self.num_envs,device=self.device)
            self.joint_vel_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
            for _ in range(10):
                self.root_quat_buffer.compute(self.scene["robot"].data.root_quat_w.clone())
                self.root_omega_buffer.compute(self.scene["robot"].data.root_ang_vel_b.clone())
                joint_pos=self.scene["robot"].data.joint_pos[:,self.joint_ids]
                self.joint_pos_buffer.compute(joint_pos.clone())
                self.joint_vel_buffer.compute(self.scene["robot"].data.joint_vel[:,self.joint_ids].clone())
            

    def load_managers(self):
        super().load_managers()
        self.observation_manager=MyObservationManager(self.cfg.observations,self)
        print("[INFO] Overload Observation Manager: MyObservationManager")

    


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
                self.joint_pos_buffer.compute(joint_pos.clone())
                self.joint_vel_buffer.compute(self.scene["robot"].data.joint_vel[:,self.joint_ids].clone())

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
                self.joint_pos_buffer.reset(reset_env_ids)
                self.joint_vel_buffer.reset(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            if self.root_quat_buffer is not None:
                for _ in range(10):
                    self.root_quat_buffer.compute(self.scene["robot"].data.root_quat_w.clone())
                    self.root_omega_buffer.compute(self.scene["robot"].data.root_ang_vel_b.clone())
                    joint_pos=self.scene["robot"].data.joint_pos[:,self.joint_ids]-self.scene["robot"].data.default_joint_pos[:,self.joint_ids]
                    self.joint_pos_buffer.compute(joint_pos.clone())
                    self.joint_vel_buffer.compute(self.scene["robot"].data.joint_vel[:,self.joint_ids].clone())

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
    
    def get_amp_obs_for_expert_trans(self):
        amp_dof_pos=self.scene["robot"].data.joint_pos[:,self.joint_ids]
        amp_dof_vel=self.scene["robot"].data.joint_vel[:,self.joint_ids]
        amp_root_lin_vel=self.scene["robot"].data.root_lin_vel_b
        amp_root_ang_vel=self.scene["robot"].data.root_ang_vel_b
        amp_left_hand_pos_b=self.scene["robot"].data.body_state_w[:,self.end_link_ids[0],:3]-self.scene["robot"].data.root_state_w[:,:3]+\
                           quat_apply(self.scene["robot"].data.body_state_w[:,self.end_link_ids[0],3:7],self.hand_offset)
        amp_left_hand_pos_b=quat_apply(quat_conjugate(self.scene["robot"].data.root_state_w[:,3:7]),amp_left_hand_pos_b)
        amp_right_hand_pos_b=self.scene["robot"].data.body_state_w[:,self.end_link_ids[1],:3]-self.scene["robot"].data.root_state_w[:,:3]+\
                           quat_apply(self.scene["robot"].data.body_state_w[:,self.end_link_ids[1],3:7],self.hand_offset)
        amp_right_hand_pos_b=quat_apply(quat_conjugate(self.scene["robot"].data.root_state_w[:,3:7]),amp_right_hand_pos_b)
        amp_left_foot_pos_b=self.scene["robot"].data.body_state_w[:,self.end_link_ids[2],:3]-self.scene["robot"].data.root_state_w[:,:3]
        amp_left_foot_pos_b=quat_apply(quat_conjugate(self.scene["robot"].data.root_state_w[:,3:7]),amp_left_foot_pos_b)
        amp_right_foot_pos_b=self.scene["robot"].data.body_state_w[:,self.end_link_ids[3],:3]-self.scene["robot"].data.root_state_w[:,:3]
        amp_right_foot_pos_b=quat_apply(quat_conjugate(self.scene["robot"].data.root_state_w[:,3:7]),amp_right_foot_pos_b)
        return torch.cat(
            (
                amp_dof_pos,
                amp_dof_vel,
                amp_root_lin_vel,
                amp_root_ang_vel,
                amp_left_hand_pos_b,
                amp_right_hand_pos_b,
                amp_left_foot_pos_b,
                amp_right_foot_pos_b,
            ),
            dim=-1,
        )