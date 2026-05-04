# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from plr_tasks.patch_locomotion.env import PLRLocomotionEnv

from plr_tasks.patch_locomotion.env_cfg import (
    PLRPatchLocomotionEnvCfg,
    PLRPatchLocomotionEnvPlayCfg,
)

from .agents.rsl_rl_cfg import AnymalDPatchPPORunnerCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Anymal-Patch-v0",
    entry_point="plr_tasks.patch_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=False,
    kwargs={
        "env_cfg_entry_point": PLRPatchLocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": AnymalDPatchPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Anymal-Patch-Play-v0",
    entry_point="plr_tasks.patch_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PLRPatchLocomotionEnvPlayCfg,
        "rsl_rl_cfg_entry_point": AnymalDPatchPPORunnerCfg,
    },
)


