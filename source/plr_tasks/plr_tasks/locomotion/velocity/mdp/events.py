


import torch
from isaaclab.envs import ManagerBasedRLEnv

from .binary_map_cfg import (
    BinaryMapGeomCfg,
    BinaryMapResetCfg,
    BinaryMapIntervalCfg,
)


def _set_map_metadata(env: ManagerBasedRLEnv, map_h: int, map_w: int, map_res: float) -> None:
    env.plr_map_origin_xy = torch.tensor(
        [-(map_w * map_res) / 2.0, -(map_h * map_res) / 2.0],
        device=env.device,
        dtype=torch.float32,
    )
    env.plr_map_resolution = float(map_res)


def _sample_rectangles_map(
    map_h: int,
    map_w: int,
    device: torch.device,
    num_rectangles: int,
    min_rect_size: int,
    max_rect_size: int,
    add_border: bool = True,
) -> torch.Tensor:
    grid = torch.ones((map_h, map_w), device=device, dtype=torch.float32)

    if add_border:
        grid[0, :] = 0.0
        grid[-1, :] = 0.0
        grid[:, 0] = 0.0
        grid[:, -1] = 0.0

    for _ in range(num_rectangles):
        rect_h = int(torch.randint(min_rect_size, max_rect_size + 1, (1,), device=device).item())
        rect_w = int(torch.randint(min_rect_size, max_rect_size + 1, (1,), device=device).item())

        max_top = max(2, map_h - rect_h)
        max_left = max(2, map_w - rect_w)

        top = int(torch.randint(1, max_top, (1,), device=device).item())
        left = int(torch.randint(1, max_left, (1,), device=device).item())

        grid[top : top + rect_h, left : left + rect_w] = 0.0

    return grid


def randomize_global_binary_map(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    map_h: int = BinaryMapGeomCfg.MAP_H,
    map_w: int = BinaryMapGeomCfg.MAP_W,
    map_res: float = BinaryMapGeomCfg.MAP_RES,
    num_rectangles_min: int = BinaryMapResetCfg.NUM_RECTANGLES_MIN,
    num_rectangles_max: int = BinaryMapResetCfg.NUM_RECTANGLES_MAX,
    min_rect_size: int = BinaryMapResetCfg.MIN_RECT_SIZE,
    max_rect_size: int = BinaryMapResetCfg.MAX_RECT_SIZE,
    add_border: bool = BinaryMapGeomCfg.ADD_BORDER,
) -> None:
    _set_map_metadata(env, map_h, map_w, map_res)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if (not hasattr(env, "plr_base_binary_map")
            or env.plr_base_binary_map.shape != (env.num_envs, map_h, map_w)):
        env.plr_base_binary_map = torch.ones((env.num_envs, map_h, map_w), device=env.device, dtype=torch.float32)

    if (not hasattr(env, "plr_dynamic_patch_map")
            or env.plr_dynamic_patch_map.shape != (env.num_envs, map_h, map_w)):
        env.plr_dynamic_patch_map = torch.ones((env.num_envs, map_h, map_w), device=env.device, dtype=torch.float32)

    for env_id in env_ids:
        idx = int(env_id.item())
        num_rectangles = int(
            torch.randint(num_rectangles_min, num_rectangles_max + 1, (1,), device=env.device).item()
        )
        env.plr_base_binary_map[idx] = _sample_rectangles_map(
            map_h=map_h,
            map_w=map_w,
            device=env.device,
            num_rectangles=num_rectangles,
            min_rect_size=min_rect_size,
            max_rect_size=max_rect_size,
            add_border=add_border,
        )
        env.plr_dynamic_patch_map[idx].fill_(1.0)

    env.plr_global_binary_map = torch.minimum(env.plr_base_binary_map, env.plr_dynamic_patch_map)

def update_dynamic_binary_patches(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    num_patches: int = BinaryMapIntervalCfg.NUM_PATCHES,
    patch_size: int = BinaryMapIntervalCfg.PATCH_SIZE,
) -> None:
    if not hasattr(env, "plr_base_binary_map"):
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    map_h = env.plr_base_binary_map.shape[1]
    map_w = env.plr_base_binary_map.shape[2]

    for env_id in env_ids:
        idx = int(env_id.item())

        # reset dynamic layer to all allowed, then stamp a few forbidden patches
        env.plr_dynamic_patch_map[idx].fill_(1.0)

        for _ in range(num_patches):
            top = int(torch.randint(0, max(1, map_h - patch_size + 1), (1,), device=env.device).item())
            left = int(torch.randint(0, max(1, map_w - patch_size + 1), (1,), device=env.device).item())
            env.plr_dynamic_patch_map[idx, top : top + patch_size, left : left + patch_size] = 0.0

    env.plr_global_binary_map = torch.minimum(env.plr_base_binary_map, env.plr_dynamic_patch_map)

