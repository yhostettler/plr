# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from plr_tasks.forward_patch_locomotion.env import PLRLocomotionEnv

from plr_tasks.forward_patch_locomotion.env_cfg import (
    PLRLocomotionForwardNoPatchEnvCfg,
    PLRLocomotionForwardNoPatchEnvPlayCfg,
    PLRLocomotionForwardPatchEnvCfg,
    PLRLocomotionForwardPatchEnvPlayCfg,
)

from .agents.rsl_rl_cfg import AnymalDForwardPatchPPORunnerCfg


##
# Register Gym environments.
##

# Baseline: same setup as patch task but without binary map — for direct comparison
gym.register(
    id="Isaac-PLR-Forward-No-Patch-Anymal-D-v0",
    entry_point="plr_tasks.forward_patch_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=False,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionForwardNoPatchEnvCfg,
        "rsl_rl_cfg_entry_point": AnymalDForwardPatchPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-PLR-Forward-No-Patch-Anymal-D-Play-v0",
    entry_point="plr_tasks.forward_patch_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionForwardNoPatchEnvPlayCfg,
        "rsl_rl_cfg_entry_point": AnymalDForwardPatchPPORunnerCfg,
    },
)

# Forward locomotion with static binary map patches
gym.register(
    id="Isaac-PLR-Forward-Patch-Anymal-D-v0",
    entry_point="plr_tasks.forward_patch_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=False,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionForwardPatchEnvCfg,
        "rsl_rl_cfg_entry_point": AnymalDForwardPatchPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-PLR-Forward-Patch-Anymal-D-Play-v0",
    entry_point="plr_tasks.forward_patch_locomotion.env:PLRLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PLRLocomotionForwardPatchEnvPlayCfg,
        "rsl_rl_cfg_entry_point": AnymalDForwardPatchPPORunnerCfg,
    },
)
