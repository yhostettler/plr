# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forbidden_patch_activation(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "forbidden_patch",
    target_weight: float = -5,
    start_step: int = 24_000,
    ramp_steps: int = 48_000,
) -> torch.Tensor:
    """Linearly ramp the forbidden-patch penalty from 0 to its full weight.

    The robot is first allowed to learn a stable gait without any forbidden-zone
    penalty.  After ``start_step`` environment steps the weight is linearly
    interpolated from 0.0 to ``target_weight`` over the next ``ramp_steps``
    steps.

    Default timing (RSL-RL, 24 steps/iter):
      - ramp begins  at iteration ~1 000  (step 24 000)
      - ramp ends    at iteration ~3 000  (step 72 000)

    Args:
        env: The RL environment.
        env_ids: Environments being reset — unused; the weight change is global.
        reward_term_name: Name of the reward term whose weight is scaled.
        target_weight: Final (fully active) weight, e.g. -0.5.
        start_step: Step at which the ramp begins (``env.common_step_counter``).
        ramp_steps: Duration of the ramp in environment steps.

    Returns:
        Scalar tensor with the current activation scale in [0, 1].
    """
    step = env.common_step_counter

    if step <= start_step:
        scale = 0.0
    elif step >= start_step + ramp_steps:
        scale = 1.0
    else:
        scale = float(step - start_step) / float(ramp_steps)

    # Patch the live weight in the reward manager so the change takes effect
    # immediately without requiring a restart.
    if reward_term_name in env.reward_manager._term_names:
        idx = env.reward_manager._term_names.index(reward_term_name)
        env.reward_manager._term_cfgs[idx].weight = scale * target_weight

    return torch.tensor(scale, device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
