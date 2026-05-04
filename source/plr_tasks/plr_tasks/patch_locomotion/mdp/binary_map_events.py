import torch
from isaaclab.envs import ManagerBasedRLEnv

from .binary_map_cfg import (
    BinaryMapGeomCfg,
    BinaryMapResetCfg,
)


def _ensure_map_metadata(
    env: ManagerBasedRLEnv,
    map_h: int,
    map_w: int,
    map_res: float,
) -> None:
    """Ensure global map metadata exists on the environment."""
    env.plr_map_origin_xy = torch.tensor(
        [
            -(map_w * map_res) / 2.0,
            -(map_h * map_res) / 2.0,
        ],
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
    add_border: bool,
) -> torch.Tensor:
    """
    Create a random binary map with rectangular forbidden regions.

    Convention:
        0 = forbidden
        1 = allowed
    """
    grid = torch.ones((map_h, map_w), device=device, dtype=torch.float32)

    if add_border:
        grid[0, :] = 0.0
        grid[-1, :] = 0.0
        grid[:, 0] = 0.0
        grid[:, -1] = 0.0

    # If border is disabled, allow rectangles anywhere.
    row_min = 1 if add_border else 0
    col_min = 1 if add_border else 0

    for _ in range(num_rectangles):
        rect_h = int(torch.randint(min_rect_size, max_rect_size + 1, (1,), device=device).item())
        rect_w = int(torch.randint(min_rect_size, max_rect_size + 1, (1,), device=device).item())

        max_top = max(row_min + 1, map_h - rect_h + 1)
        max_left = max(col_min + 1, map_w - rect_w + 1)

        top = int(torch.randint(row_min, max_top, (1,), device=device).item())
        left = int(torch.randint(col_min, max_left, (1,), device=device).item())

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
    """
    Randomize a global binary map for the selected environments.

    Convention:
        0 = forbidden
        1 = allowed
    """
    _ensure_map_metadata(env, map_h, map_w, map_res)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if (
        not hasattr(env, "plr_global_binary_map")
        or env.plr_global_binary_map.shape != (env.num_envs, map_h, map_w)
    ):
        env.plr_global_binary_map = torch.ones(
            (env.num_envs, map_h, map_w),
            device=env.device,
            dtype=torch.float32,
        )

    for env_id in env_ids.tolist():
        num_rectangles = int(
            torch.randint(num_rectangles_min, num_rectangles_max + 1, (1,), device=env.device).item()
        )
        env.plr_global_binary_map[env_id] = _sample_rectangles_map(
            map_h=map_h,
            map_w=map_w,
            device=env.device,
            num_rectangles=num_rectangles,
            min_rect_size=min_rect_size,
            max_rect_size=max_rect_size,
            add_border=add_border,
        )
