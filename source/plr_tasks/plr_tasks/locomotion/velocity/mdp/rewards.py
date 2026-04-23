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
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, max_air_time: float
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
    last_air_time = torch.clamp(last_air_time, max=max_air_time)
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
) -> torch.Tensor:
    """Penalize the robot when any foot makes contact with a forbidden map cell.

    The global binary map uses the convention:
        0.0 = forbidden  (obstacle / restricted zone)
        1.0 = allowed    (free space)

    A foot counts as "stepping" only when its ground-contact force exceeds
    ``contact_threshold``.  Hovering feet are ignored even if their XY
    projection falls over a forbidden cell.

    This function returns 1.0 for every environment where at least one foot
    is both in contact with the ground AND projected onto a forbidden map cell,
    and 0.0 otherwise.  The actual penalty magnitude is controlled by the
    negative weight in RewardTermCfg (e.g. weight=-0.5).

    Args:
        env: The RL environment, expected to carry the binary map attributes
             set by the ``randomize_global_binary_map`` event:
               - ``env.plr_global_binary_map``  (num_envs, H, W) float tensor
               - ``env.plr_map_origin_xy``       (2,) tensor, world-frame XY of
                                                  map cell (0, 0) in metres
               - ``env.plr_map_resolution``      float, metres per cell
        sensor_cfg: Contact sensor config with ``body_names=".*FOOT"`` so that
                    ``sensor_cfg.body_ids`` resolves to the foot contact bodies.
                    Must be passed through ``RewardTermCfg.params`` for the
                    manager framework to resolve body IDs.
        asset_cfg: Articulation config with ``body_names=".*FOOT"`` so that
                   ``asset_cfg.body_ids`` resolves to the foot rigid bodies.
                   Must be passed through ``RewardTermCfg.params``.
        contact_threshold: Minimum net contact force magnitude (N) for a foot
                           to be considered in contact with the ground.

    Returns:
        Tensor of shape (num_envs,) with values in {0.0, 1.0}.

    Note:
        Before the first reset event the map does not yet exist on ``env``.
        The early-out guard returns zeros in that case so the reward manager
        does not crash during the very first step.
    """
    # The binary map is written onto env by the reset event (randomize_global_binary_map).
    # On the very first call before any reset has run, the attribute does not exist yet.
    # Return an all-zero tensor so the reward manager does not crash on that first step.
    if not hasattr(env, "plr_global_binary_map"):
        return torch.zeros(env.num_envs, device=env.device)

    # -------------------------------------------------------------------------
    # Step 1 — Build a per-foot ground-contact mask
    # -------------------------------------------------------------------------

    # Retrieve the contact sensor object from the scene by name.
    # Type annotation lets IDEs resolve the ContactSensor API.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # net_forces_w_history has shape (num_envs, history_len, num_bodies, 3).
    # We index dim=2 with sensor_cfg.body_ids to keep only the foot bodies,
    # giving (num_envs, history_len, num_feet, 3).
    # .norm(dim=-1) collapses the xyz force vector to a scalar magnitude,
    # giving (num_envs, history_len, num_feet).
    # .max(dim=1)[0] takes the maximum magnitude over the history window,
    # giving (num_envs, num_feet).
    # Using the max (rather than the current frame only) means a foot that
    # touched down one or two physics sub-steps ago is still counted as active.
    contact_force_norm = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )  # (num_envs, num_feet)

    # Boolean mask: True where the contact force exceeds the threshold (N).
    # Feet that are swinging or hovering have near-zero force and are False here.
    in_contact = contact_force_norm > contact_threshold  # (num_envs, num_feet)

    # -------------------------------------------------------------------------
    # Step 2 — Collect world-frame XY positions of all foot bodies
    # -------------------------------------------------------------------------

    # body_pos_w has shape (num_envs, num_bodies, 3) in world frame (metres).
    # Indexing dim=1 with asset_cfg.body_ids selects only the foot links,
    # giving (num_envs, num_feet, 3).
    # Slicing :2 on the last dim drops Z; we only need the ground-plane XY.
    asset = env.scene[asset_cfg.name]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # (num_envs, num_feet, 2)
    num_feet = foot_xy.shape[1]                                  # e.g. 4 for a quadruped

    # -------------------------------------------------------------------------
    # Step 3 — Convert world-frame XY to integer map pixel indices
    # -------------------------------------------------------------------------

    # plr_map_origin_xy holds the world-frame XY coordinate (metres) that
    # corresponds to pixel (row=0, col=0) in plr_global_binary_map.
    origin_x = env.plr_map_origin_xy[0]   # world X that maps to column 0
    origin_y = env.plr_map_origin_xy[1]   # world Y that maps to row 0

    # plr_map_resolution is the side length of one pixel in metres (e.g. 0.1 m).
    map_res = float(env.plr_map_resolution)

    # Map dimensions in pixels, read from the tensor shape rather than the cfg
    # to stay correct even if the map is ever rebuilt at a different size.
    map_h, map_w = env.plr_global_binary_map.shape[1], env.plr_global_binary_map.shape[2]

    # Subtract the map origin to get the foot position relative to pixel (0,0),
    # then divide by map_res to convert metres to fractional pixel coordinates,
    # then .long() truncates to the integer pixel index (floor towards zero).
    # world X increases in the column direction; world Y in the row direction —
    # this mirrors the convention used in binary_map_local (observations.py).
    # .clamp() keeps the index inside [0, size-1] so a foot that has fallen or
    # walked outside the map boundary never causes an out-of-bounds index error.
    col = ((foot_xy[:, :, 0] - origin_x) / map_res).long().clamp(0, map_w - 1)  # (num_envs, num_feet)
    row = ((foot_xy[:, :, 1] - origin_y) / map_res).long().clamp(0, map_h - 1)  # (num_envs, num_feet)

    # -------------------------------------------------------------------------
    # Step 4 — Look up the map value for every foot in every environment
    # -------------------------------------------------------------------------

    # We need a matching env index for each (row, col) entry so we can do
    # 3D fancy indexing: map[env_i, row_i, col_i].
    # torch.arange gives (num_envs,); .unsqueeze(1) makes it (num_envs, 1);
    # .expand(-1, num_feet) broadcasts it to (num_envs, num_feet) without
    # copying memory (zero-copy view), which is cheaper than repeat_interleave.
    env_ids = torch.arange(env.num_envs, device=env.device).unsqueeze(1).expand(-1, num_feet)

    # Fancy-index the global map with three (num_envs, num_feet) tensors.
    # Each entry map[env_ids[i,j], row[i,j], col[i,j]] is the map value for
    # foot j of environment i — either 0.0 (forbidden) or 1.0 (allowed).
    map_values = env.plr_global_binary_map[env_ids, row, col]  # (num_envs, num_feet)

    # -------------------------------------------------------------------------
    # Step 5 — Combine map and contact masks; return the per-env penalty signal
    # -------------------------------------------------------------------------

    # A foot is penalised only when BOTH conditions hold simultaneously:
    #   (a) map_values < 0.5  →  the cell is forbidden (value 0.0).
    #       The 0.5 threshold is robust to any float rounding in the map tensor.
    #   (b) in_contact is True  →  the foot is actually pressing on the ground.
    # The bitwise AND combines the two boolean (num_envs, num_feet) masks.
    forbidden_and_contact = (map_values < 0.5) & in_contact  # (num_envs, num_feet)

    # .any(dim=1) reduces over the foot dimension: True if at least one foot
    # of that environment satisfies both conditions.
    # .float() converts True/False to 1.0/0.0 for the reward computation.
    # The RewardTermCfg weight (e.g. -0.5) is applied by the reward manager
    # after this function returns.
    return forbidden_and_contact.any(dim=1).float()  # (num_envs,)


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
