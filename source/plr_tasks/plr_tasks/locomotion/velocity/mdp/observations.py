import torch
from isaaclab.envs import ManagerBasedRLEnv

from .binary_map_cfg import BinaryMapGeomCfg, BinaryMapEgoCfg




def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion in (w, x, y, z) format."""
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _ensure_bootstrap_global_binary_map(env: ManagerBasedRLEnv) -> None:

    if not hasattr(env, "plr_map_origin_xy"):
        env.plr_map_origin_xy = torch.tensor(
            [-(BinaryMapGeomCfg.MAP_W * BinaryMapGeomCfg.MAP_RES) / 2.0,
             -(BinaryMapGeomCfg.MAP_H * BinaryMapGeomCfg.MAP_RES) / 2.0],
            device=env.device,
            dtype=torch.float32,
        )
        env.plr_map_resolution = float(BinaryMapGeomCfg.MAP_RES)
    if not hasattr(env, "plr_map_resolution"):
        env.plr_map_resolution = float(BinaryMapGeomCfg.MAP_RES)

    if not hasattr(env, "plr_base_binary_map"):
        base_map = torch.ones(
            (BinaryMapGeomCfg.MAP_H, BinaryMapGeomCfg.MAP_W),
            device=env.device,
            dtype=torch.float32,
        )

        if BinaryMapGeomCfg.ADD_BORDER:
            base_map[0, :] = 0.0
            base_map[-1, :] = 0.0
            base_map[:, 0] = 0.0
            base_map[:, -1] = 0.0

        env.plr_base_binary_map = base_map.unsqueeze(0).repeat(env.num_envs, 1, 1)

    if not hasattr(env, "plr_dynamic_patch_map"):
        env.plr_dynamic_patch_map = torch.ones_like(env.plr_base_binary_map)

    env.plr_global_binary_map = torch.minimum(env.plr_base_binary_map, env.plr_dynamic_patch_map)






def binary_map_2x2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a 2x2 egocentric binary map sampled from the global map.

    Order:
        [front_left, front_right, rear_left, rear_right]

    Convention:
        0 = forbidden
        1 = allowed
    """
    _ensure_bootstrap_global_binary_map(env)

    global_map = env.plr_global_binary_map
    map_h = global_map.shape[1]
    map_w = global_map.shape[2]

    robot = env.scene["robot"]

    # robot pose in world/env frame
    root_xy = robot.data.root_pos_w[:, :2]      # (num_envs, 2)
    root_quat = robot.data.root_quat_w          # (num_envs, 4), wxyz
    yaw = _yaw_from_quat_wxyz(root_quat)        # (num_envs,)

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    # [front_left, front_right, rear_left, rear_right]
    local_offsets = torch.tensor(
        [
            [ BinaryMapEgoCfg.EGO_HALF_SPAN_X,  BinaryMapEgoCfg.EGO_HALF_SPAN_Y],
            [ BinaryMapEgoCfg.EGO_HALF_SPAN_X, -BinaryMapEgoCfg.EGO_HALF_SPAN_Y],
            [-BinaryMapEgoCfg.EGO_HALF_SPAN_X,  BinaryMapEgoCfg.EGO_HALF_SPAN_Y],
            [-BinaryMapEgoCfg.EGO_HALF_SPAN_X, -BinaryMapEgoCfg.EGO_HALF_SPAN_Y],
        ],
        device=env.device,
        dtype=torch.float32,
    )
    local_x = local_offsets[:, 0].unsqueeze(0)   # (1, 4)
    local_y = local_offsets[:, 1].unsqueeze(0)   # (1, 4)

    # rotate local sample points into world/env frame
    sample_x = root_xy[:, 0:1] + local_x * c.unsqueeze(1) - local_y * s.unsqueeze(1)
    sample_y = root_xy[:, 1:2] + local_x * s.unsqueeze(1) + local_y * c.unsqueeze(1)

    origin_x = env.plr_map_origin_xy[0]
    origin_y = env.plr_map_origin_xy[1]
    res = float(env.plr_map_resolution)

    # world -> grid
    cols = torch.floor((sample_x - origin_x) / res).long()
    rows = torch.floor((sample_y - origin_y) / res).long()

    valid = (
        (rows >= 0) & (rows < map_h) &
        (cols >= 0) & (cols < map_w)
    )

    # outside map = forbidden
    out = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)

    env_ids = torch.arange(env.num_envs, device=env.device).unsqueeze(1).expand(-1, 4)
    out[valid] = global_map[env_ids[valid], rows[valid], cols[valid]]

    # store for visualization/debugging
    env.plr_last_rows = rows
    env.plr_last_cols = cols

    return out

