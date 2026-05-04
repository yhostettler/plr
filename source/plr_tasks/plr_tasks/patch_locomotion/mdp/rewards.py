# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# reward for tracking of linear velocity around x,y-axis (EMA)
def track_ema_lin_vel_xy(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute the error
    lin_vel_error = torch.sum(
        torch.square((env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2])),
        dim=1,
    )

    ema_error_lin_vel_xy = env.ema_manager.compute_ema_error_lin_vel_xy(lin_vel_error)

    # return torch.clamp((1-(ema_error_lin_vel_xy/env.ema_manager._epsilon)),min=0)
    # return 1/(1+ema_error_lin_vel_xy)
    return torch.exp(-ema_error_lin_vel_xy / std**2)


def forbidden_patch_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT"),
    contact_threshold: float = 1.0,
    base_penalty_scale: float = 0.1,
) -> torch.Tensor:
    """Penalize the robot for foot contact on or base position over a forbidden map cell.

    Returns a per-env scalar:
        - 1.0 if any foot is in contact with a forbidden cell
        - ``base_penalty_scale`` if the base XY is over a forbidden cell (but no foot contact)
        - Both can stack if both conditions hold simultaneously.

    ``base_penalty_scale`` is intentionally < 1.0 because the base hovers over a region
    for many consecutive steps, so it accumulates faster than a transient foot contact.
    """
    if not hasattr(env, "plr_global_binary_map"):  # map not yet created by reset event
        return torch.zeros(env.num_envs, device=env.device)  # safe zero penalty before first reset

    asset = env.scene[asset_cfg.name]  # articulation handle (robot)
    origin_x = env.plr_map_origin_xy[0]  # world X of map column 0
    origin_y = env.plr_map_origin_xy[1]  # world Y of map row 0
    map_res = float(env.plr_map_resolution)  # metres per cell
    map_h, map_w = env.plr_global_binary_map.shape[0], env.plr_global_binary_map.shape[1]  # map size in cells

    # ------------------------------------------------------------------
    # Foot-contact penalty
    # ------------------------------------------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # contact sensor handle
    contact_force_norm = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (num_envs, hist, num_feet, 3)
        .norm(dim=-1)  # scalar force magnitude per foot per history step → (num_envs, hist, num_feet)
        .max(dim=1)[0]  # max over history window → (num_envs, num_feet)
    )
    in_contact = contact_force_norm > contact_threshold  # True where foot is pressing the ground (num_envs, num_feet)

    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # world XY of each foot (num_envs, num_feet, 2)
    foot_col = ((foot_xy[:, :, 0] - origin_x) / map_res).long().clamp(0, map_w - 1)  # foot world X → map column index
    foot_row = ((foot_xy[:, :, 1] - origin_y) / map_res).long().clamp(0, map_h - 1)  # foot world Y → map row index
    foot_map_values = env.plr_global_binary_map[foot_row, foot_col]  # map cell value under each foot (num_envs, num_feet)

    foot_penalty = ((foot_map_values < 0.5) & in_contact).any(dim=1).float()  # 1.0 if any foot contacts a forbidden cell

    # ------------------------------------------------------------------
    # Base-over-forbidden penalty (weighted lower — persists many steps)
    # ------------------------------------------------------------------
    base_xy = asset.data.root_pos_w[:, :2]  # world XY of the robot base (num_envs, 2)
    base_col = ((base_xy[:, 0] - origin_x) / map_res).long().clamp(0, map_w - 1)  # base world X → map column index
    base_row = ((base_xy[:, 1] - origin_y) / map_res).long().clamp(0, map_h - 1)  # base world Y → map row index
    base_map_values = env.plr_global_binary_map[base_row, base_col]  # map cell value under the base (num_envs,)

    base_penalty = (base_map_values < 0.5).float() * base_penalty_scale  # base_penalty_scale if base is over forbidden cell

    return foot_penalty + base_penalty  # combined penalty; overall magnitude set by RewardTermCfg weight


# reward for tracking of angular velocity around z-axis (EMA)
def track_ema_ang_vel_z(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])

    ema_error_ang_vel_z = env.ema_manager.compute_ema_error_ang_vel_z(ang_vel_error)

    # return torch.clamp((1- (ema_error_ang_vel_z/env.ema_manager._epsilon)),min=0)
    # return 1/(1+ema_error_ang_vel_z)
    return torch.exp(-ema_error_ang_vel_z / std**2)
