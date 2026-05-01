# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from plr_tasks.terrain_locomotion.env import PLRLocomotionEnv

from plr_tasks.terrain_locomotion.env_cfg import (
    PLRLocomotionEnvCfg,
    PLRLocomotionEnvPlayCfg,
    PLRLocomotionEasyEnvCfg,
    PLRLocomotionEasyEnvPlayCfg,
)

from .agents.rsl_rl_cfg import (
    AnymalDFlatPPORunnerCfg
)
    

##
# Register Gym environments.
##

gym.register(
    id="Isaac-PLR-Velocity-Flat-Anymal-D-v0",
    entry_point="plr_tasks.terrain_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=False,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": AnymalDFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-PLR-Velocity-Flat-Anymal-D-Play-v0",
    entry_point="plr_tasks.terrain_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionEnvPlayCfg,
        "rsl_rl_cfg_entry_point": AnymalDFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-PLR-Velocity-Easy-Anymal-D-v0",
    entry_point="plr_tasks.terrain_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=False,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionEasyEnvCfg,
        "rsl_rl_cfg_entry_point": AnymalDFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-PLR-Velocity-Easy-Anymal-D-Play-v0",
    entry_point="plr_tasks.terrain_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionEasyEnvPlayCfg,
        "rsl_rl_cfg_entry_point": AnymalDFlatPPORunnerCfg,
    },
)

