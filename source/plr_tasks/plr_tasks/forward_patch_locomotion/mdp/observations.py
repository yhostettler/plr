import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .binary_map_cfg import (
    BinaryMapGeomCfg,
    BinaryMapLocalCfg,
)


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion in (w, x, y, z) format."""
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

# updated to only one global map for all environments
def _ensure_bootstrap_global_binary_map(env: ManagerBasedRLEnv) -> None:
    """Create a default global binary map if events have not populated it yet."""
    if not hasattr(env, "plr_map_origin_xy"):
        env.plr_map_origin_xy = torch.tensor(
            [
                -(BinaryMapGeomCfg.MAP_W * BinaryMapGeomCfg.MAP_RES) / 2.0,
                -(BinaryMapGeomCfg.MAP_H * BinaryMapGeomCfg.MAP_RES) / 2.0,
            ],
            device=env.device,
            dtype=torch.float32,
        )

    if not hasattr(env, "plr_map_resolution"):
        env.plr_map_resolution = float(BinaryMapGeomCfg.MAP_RES)

    if not hasattr(env, "plr_global_binary_map"):
        grid = torch.ones(
            (BinaryMapGeomCfg.MAP_H, BinaryMapGeomCfg.MAP_W),
            device=env.device,
            dtype=torch.float32,
        )
        if BinaryMapGeomCfg.ADD_BORDER:
            grid[0, :] = 0.0
            grid[-1, :] = 0.0
            grid[:, 0] = 0.0
            grid[:, -1] = 0.0
        env.plr_global_binary_map = grid


def _get_local_grid_xy(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return cached local robot-frame grid cell centers.

    Convention:
    - +x_local = forward
    - +y_local = right
    - row 0 is most-forward
    - col 0 is most-left

    For even sizes like 64x64, the robot lies between the four middle cells.
    """
    cache_key = (
        BinaryMapLocalCfg.LOCAL_H,
        BinaryMapLocalCfg.LOCAL_W,
        BinaryMapLocalCfg.LOCAL_RES,
        str(env.device),
    )

    if getattr(env, "_plr_local_grid_cache_key", None) == cache_key:
        return env._plr_local_x, env._plr_local_y

    local_h = BinaryMapLocalCfg.LOCAL_H
    local_w = BinaryMapLocalCfg.LOCAL_W
    local_res = BinaryMapLocalCfg.LOCAL_RES

    row_coords = (
        (local_h - 1) / 2.0 - torch.arange(local_h, device=env.device, dtype=torch.float32)
    ) * local_res
    col_coords = (
        torch.arange(local_w, device=env.device, dtype=torch.float32) - (local_w - 1) / 2.0
    ) * local_res

    x_local = row_coords.view(local_h, 1).expand(local_h, local_w)
    y_local = col_coords.view(1, local_w).expand(local_h, local_w)

    env._plr_local_x = x_local
    env._plr_local_y = y_local
    env._plr_local_grid_cache_key = cache_key
    return x_local, y_local


def binary_map_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Return a robot-aligned local binary map sampled from the global map.

    Convention:
        0 = forbidden
        1 = allowed

    Output shape:
        - (num_envs, 4096) if BinaryMapLocalCfg.FLATTEN_OUTPUT = True
        - (num_envs, 64, 64) otherwise
    """
    _ensure_bootstrap_global_binary_map(env)

    # Shared (H,W) map — broadcast to (B,H,W) for grid_sample
    global_map = env.plr_global_binary_map.float().unsqueeze(0).expand(env.num_envs, -1, -1)
    map_h = global_map.shape[1]
    map_w = global_map.shape[2]

    robot = env.scene["robot"]
    root_xy = robot.data.root_pos_w[:, :2]   # meters
    root_quat = robot.data.root_quat_w
    yaw = _yaw_from_quat_wxyz(root_quat)

    x_local, y_local = _get_local_grid_xy(env)
    x_local = x_local.unsqueeze(0).expand(env.num_envs, -1, -1)
    y_local = y_local.unsqueeze(0).expand(env.num_envs, -1, -1)

    c = torch.cos(yaw).view(env.num_envs, 1, 1)
    s = torch.sin(yaw).view(env.num_envs, 1, 1)

    px = root_xy[:, 0].view(env.num_envs, 1, 1)
    py = root_xy[:, 1].view(env.num_envs, 1, 1)

    # Robot frame -> world frame
    wx = px + x_local * c + y_local * s
    wy = py - x_local * s + y_local * c

    origin_x = env.plr_map_origin_xy[0]
    origin_y = env.plr_map_origin_xy[1]
    map_res = float(env.plr_map_resolution)

    # World meters -> global map fractional indices
    cols = (wx - origin_x) / map_res
    rows = (wy - origin_y) / map_res

    # Fractional indices -> normalized grid_sample coordinates
    grid_x = (cols / (map_w - 1)) * 2.0 - 1.0
    grid_y = (rows / (map_h - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)   # (B,64,64,2)

    inp = global_map.unsqueeze(1)  # (B,1,H,W)
    out = F.grid_sample(
        inp,
        grid,
        mode=BinaryMapLocalCfg.SAMPLE_MODE,
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)  # (B,64,64)

    # Optional custom out-of-bounds value.
    if BinaryMapLocalCfg.OUT_OF_BOUNDS_VALUE != 0.0:
        oob = (grid_x.abs() > 1.0) | (grid_y.abs() > 1.0)
        out[oob] = float(BinaryMapLocalCfg.OUT_OF_BOUNDS_VALUE)

    # Save for debugging / visualization if needed.
    env.plr_last_local_rows = rows
    env.plr_last_local_cols = cols

    if BinaryMapLocalCfg.FLATTEN_OUTPUT:
        return out.view(env.num_envs, -1)
    return out


def base_lin_ang_vel_err_ema(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute ema tensor (lin_vel_x_ema, lin_vel_y_ema, ang_vel_z_ema)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_xy = asset.data.root_lin_vel_b[:, :2]
    ang_vel_z = asset.data.root_ang_vel_b[:, -1].unsqueeze(1)
    return env.ema_manager.compute_ema_signal(torch.cat((lin_vel_xy, ang_vel_z), dim=1))
