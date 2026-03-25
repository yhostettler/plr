import gymnasium as gym

from . import plr_nav_env_cfg
from .agents import rsl_rl_cfg

gym.register(
    id="PLR-Nav-PPO-B2W-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": plr_nav_env_cfg.PLRB2WNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_cfg.PLRB2WNavPPORunnerCfg,
    },
)

gym.register(
    id="PLR-Nav-PPO-B2W-Play-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": plr_nav_env_cfg.PLRB2WNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_cfg.PLRB2WNavPPORunnerCfg,
    },
)

gym.register(
    id="PLR-Nav-PPO-B2W-Dev-v0",
    entry_point="isaaclab_nav_task.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": plr_nav_env_cfg.PLRB2WNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": rsl_rl_cfg.PLRB2WNavPPORunnerDevCfg,
    },
)
