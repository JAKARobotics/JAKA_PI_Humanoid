import gymnasium as gym

gym.register(
    id="Jaka-Pi-Mimic-Dance",
    entry_point="jaka_rl_lab.tasks.mimic.mdp:MyRLEnv",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:JakaPiMimicEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:JakaPiMimicPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"jaka_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)