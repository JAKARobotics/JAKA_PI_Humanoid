import gymnasium as gym
import os
ISAAC_K1L_MINI_LOCO_DIR = os.path.abspath(os.path.dirname(__file__))

gym.register(
    id="K1L-mini-loco",
    entry_point="jaka_rl_lab.tasks.locomotion.mdp:MyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"jaka_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="K1L-mini-loco-Sym",
    entry_point="jaka_rl_lab.tasks.locomotion.mdp:MyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"jaka_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:MiniSymPPORunnerCfg",
    },
)


gym.register(
    id="K1L-mini-loco-SymAmp",
    entry_point="jaka_rl_lab.tasks.locomotion.mdp:MyRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"jaka_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:MiniSymAmpPPORunnerCfg",
    },
)