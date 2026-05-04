# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_on_forbidden_patch(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when any foot makes contact with a forbidden map cell.

    Uses the hard binary map (plr_global_binary_map, 0=forbidden / 1=allowed) rather than the
    soft Gaussian penalty field, so the termination fires only when a foot is exactly on a
    forbidden cell — not merely nearby.

    Returns a bool tensor of shape (num_envs,): True where the episode should end.
    """
    if not hasattr(env, "plr_global_binary_map"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force_norm = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )  # (num_envs, num_feet)
    in_contact = contact_force_norm > contact_threshold  # (num_envs, num_feet)

    asset = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # (num_envs, num_feet, 2)
    num_feet = foot_xy.shape[1]

    origin_x = env.plr_map_origin_xy[0]
    origin_y = env.plr_map_origin_xy[1]
    map_res = float(env.plr_map_resolution)
    map_h, map_w = env.plr_global_binary_map.shape[0], env.plr_global_binary_map.shape[1]

    col = ((foot_xy[:, :, 0] - origin_x) / map_res).long().clamp(0, map_w - 1)
    row = ((foot_xy[:, :, 1] - origin_y) / map_res).long().clamp(0, map_h - 1)

    map_values = env.plr_global_binary_map[row, col]  # (num_envs, num_feet)

    forbidden_and_contact = (map_values < 0.5) & in_contact  # (num_envs, num_feet)
    return forbidden_and_contact.any(dim=1)  # (num_envs,)


def base_over_forbidden_patch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot's base is directly over a forbidden map cell.

    Uses the hard binary map (plr_global_binary_map) so the termination fires only when the
    base XY projection lands exactly on a forbidden cell, not merely nearby.

    Returns a bool tensor of shape (num_envs,): True where the episode should end.
    """
    if not hasattr(env, "plr_global_binary_map"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    asset = env.scene[asset_cfg.name]
    base_xy = asset.data.root_pos_w[:, :2]  # (num_envs, 2)

    origin_x = env.plr_map_origin_xy[0]
    origin_y = env.plr_map_origin_xy[1]
    map_res = float(env.plr_map_resolution)
    map_h, map_w = env.plr_global_binary_map.shape[0], env.plr_global_binary_map.shape[1]

    col = ((base_xy[:, 0] - origin_x) / map_res).long().clamp(0, map_w - 1)
    row = ((base_xy[:, 1] - origin_y) / map_res).long().clamp(0, map_h - 1)

    map_values = env.plr_global_binary_map[row, col]  # (num_envs,)
    return map_values < 0.5  # (num_envs,)


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
