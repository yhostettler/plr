import torch
from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------
# static global map parameters
# -----------------------------
PLR_MAP_H = 64
PLR_MAP_W = 64
PLR_MAP_RES = 0.10  # [m / cell]

# distance from robot center to each of the 2x2 sample cells
PLR_EGO_HALF_SPAN_X = 0.15
PLR_EGO_HALF_SPAN_Y = 0.15


def _ensure_global_binary_map(env: ManagerBasedRLEnv) -> None:
    """Create one static global binary map per env clone.

    Convention:
        0 = forbidden
        1 = allowed
    """
    if hasattr(env, "plr_global_binary_map"):
        return

    device = env.device
    dtype = torch.float32

    # one static base map
    base_map = torch.ones((PLR_MAP_H, PLR_MAP_W), device=device, dtype=dtype)

    # optional: make borders forbidden
    base_map[0, :] = 0.0
    base_map[-1, :] = 0.0
    base_map[:, 0] = 0.0
    base_map[:, -1] = 0.0

    # optional: add one static forbidden block in the middle
    base_map[26:38, 26:38] = 0.0

    # repeat for all parallel envs
    env.plr_global_binary_map = base_map.unsqueeze(0).repeat(env.num_envs, 1, 1)

    # map origin in env frame: lower-left corner
    env.plr_map_origin_xy = torch.tensor(
        [-(PLR_MAP_W * PLR_MAP_RES) / 2.0, -(PLR_MAP_H * PLR_MAP_RES) / 2.0],
        device=device,
        dtype=dtype,
    )
    env.plr_map_resolution = PLR_MAP_RES


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion in (w, x, y, z) order."""
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def binary_map_global(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the full global binary map for every env.

    Shape:
        (num_envs, H, W)
    """
    _ensure_global_binary_map(env)
    return env.plr_global_binary_map


def binary_map_2x2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a 2x2 egocentric binary map sampled from the global map.

    Order:
        [front_left, front_right, rear_left, rear_right]

    Convention:
        0 = forbidden
        1 = allowed

    Assumption:
        robot local frame uses x forward, y left
    """
    _ensure_global_binary_map(env)

    robot = env.scene["robot"]

    # robot pose in env frame
    root_xy = robot.data.root_pos_w[:, :2]       # (num_envs, 2)
    root_quat = robot.data.root_quat_w           # (num_envs, 4), (w, x, y, z)
    yaw = _yaw_from_quat_wxyz(root_quat)         # (num_envs,)

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    # local egocentric sample points:
    # [front_left, front_right, rear_left, rear_right]
    local_offsets = torch.tensor(
        [
            [ PLR_EGO_HALF_SPAN_X,  PLR_EGO_HALF_SPAN_Y],
            [ PLR_EGO_HALF_SPAN_X, -PLR_EGO_HALF_SPAN_Y],
            [-PLR_EGO_HALF_SPAN_X,  PLR_EGO_HALF_SPAN_Y],
            [-PLR_EGO_HALF_SPAN_X, -PLR_EGO_HALF_SPAN_Y],
        ],
        device=env.device,
        dtype=torch.float32,
    )  # (4, 2)

    local_x = local_offsets[:, 0].unsqueeze(0)   # (1, 4)
    local_y = local_offsets[:, 1].unsqueeze(0)   # (1, 4)

    # rotate local offsets into env frame
    sample_x = root_xy[:, 0:1] + local_x * c.unsqueeze(1) - local_y * s.unsqueeze(1)
    sample_y = root_xy[:, 1:2] + local_x * s.unsqueeze(1) + local_y * c.unsqueeze(1)

    origin_x = env.plr_map_origin_xy[0]
    origin_y = env.plr_map_origin_xy[1]
    res = env.plr_map_resolution

    # world/env coordinates -> grid indices
    cols = torch.floor((sample_x - origin_x) / res).long()
    rows = torch.floor((sample_y - origin_y) / res).long()

    valid = (
        (rows >= 0) & (rows < env.plr_global_binary_map.shape[1]) &
        (cols >= 0) & (cols < env.plr_global_binary_map.shape[2])
    )

    # default outside map = forbidden
    out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)

    env_ids = torch.arange(env.num_envs, device=env.device).unsqueeze(1).expand(-1, 4)

    out[valid] = env.plr_global_binary_map[env_ids[valid], rows[valid], cols[valid]]
    return outfrom pathlib import Path

import torch
from isaaclab.envs import ManagerBasedRLEnv


def binary_map_2x2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the stored 2x2 egocentric binary map for every environment.

    Order:
        [front_left, front_right, rear_left, rear_right]

    Convention:
        0 = forbidden
        1 = allowed
    """
    if not hasattr(env, "plr_binary_map_2x2"):
        pattern = torch.tensor([0.0, 1.0, 1.0, 0.0], device=env.device, dtype=torch.float32)
        env.plr_binary_map_2x2 = pattern.unsqueeze(0).repeat(env.num_envs, 1)

    return env.plr_binary_map_2x2


