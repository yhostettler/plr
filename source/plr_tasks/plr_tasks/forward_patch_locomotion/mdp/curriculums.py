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

# added (Leon) -> patch penalty curriculum
def patch_penalty_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_weight_base: float = -0.02,
    end_weight_base: float = -0.5,
    start_weight_foot: float = -0.02,
    end_weight_foot: float = -0.2,
    ramp_end_steps: int = 48_000,
) -> torch.Tensor:
    """Linearly ramp forbidden-patch penalty weights over training.

    Uses env.common_step_counter as a proxy for training progress.
    With num_steps_per_env=24, ramp_end_steps=48_000 ≈ 2000 RSL-RL iterations.

    Returns the current ramp fraction in [0, 1] for logging.
    """
    t = min(float(env.common_step_counter) / max(ramp_end_steps, 1), 1.0)

    env.reward_manager.get_term_cfg("base_over_forbidden_patch").weight = (
        start_weight_base + t * (end_weight_base - start_weight_base)
    )
    env.reward_manager.get_term_cfg("forbidden_patch").weight = (
        start_weight_foot + t * (end_weight_foot - start_weight_foot)
    )

    return torch.tensor(t, device=env.device)
