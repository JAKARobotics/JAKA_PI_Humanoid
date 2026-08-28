import gymnasium as gym
import os
ISAAC_JAKA_PI_LOCO_DIR = os.path.abspath(os.path.dirname(__file__))

gym.register(
    id="Jaka-Pi-Loco",
    entry_point="jaka_rl_lab.tasks.locomotion.mdp:MyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"jaka_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Jaka-Pi-Loco-Sym",
    entry_point="jaka_rl_lab.tasks.locomotion.mdp:MyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"jaka_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:SymPPORunnerCfg",
    },
)


gym.register(
    id="Jaka-Pi-Loco-SymAmp",
    entry_point="jaka_rl_lab.tasks.locomotion.mdp:MyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"jaka_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:SymAmpPPORunnerCfg",
    },
)