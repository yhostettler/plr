# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

##############################################################################################################
# MDPO

# gym.register(
#     id="Isaac-Nav-MDPO-ANYmal-D-v0",
#     entry_point="plr_tasks.navigation:NavigationEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.ANYmalDNavigationEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ANYmalDNavMDPORunnerCfg,
#     },
# )

# gym.register(
#     id="Isaac-Nav-MDPO-ANYmal-D-Play-v0",
#     entry_point="plr_tasks.navigation:NavigationEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.ANYmalDNavigationEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ANYmalDNavMDPORunnerCfg,
#     },
# )

# gym.register(
#     id="Isaac-Nav-MDPO-ANYmal-D-Dev-v0",
#     entry_point="plr_tasks.navigation:NavigationEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.ANYmalDNavigationEnvCfg_DEV,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ANYmalDNavMDPORunnerDevCfg,
#     },
# )

######################################################################################
# PPO

gym.register(
    id="Isaac-Nav-PPO-ANYmal-D-v0",
    entry_point="plr_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.ANYmalDNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ANYmalDNavPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-ANYmal-D-Play-v0",
    entry_point="plr_tasks.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.ANYmalDNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ANYmalDNavPPORunnerCfg,
    },
)

# gym.register(
#     id="Isaac-Nav-PPO-ANYmal-D-Dev-v0",
#     entry_point="plr_tasks.navigation:NavigationEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": navigation_env_cfg.ANYmalDNavigationEnvCfg_DEV,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ANYmalDNavPPORunnerDevCfg,
#     },
# )
