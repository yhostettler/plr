import torch
from isaaclab.envs import ManagerBasedRLEnv


def _ensure_map_metadata(
    env: ManagerBasedRLEnv,
    map_h: int,
    map_w: int,
    map_res: float,
) -> None:
    """Ensure global map metadata exists on the environment."""
    if not hasattr(env, "plr_map_origin_xy"):
        env.plr_map_origin_xy = torch.tensor(
            [-(map_w * map_res) / 2.0, -(map_h * map_res) / 2.0],
            device=env.device,
            dtype=torch.float32,
        )
    if not hasattr(env, "plr_map_resolution"):
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
    """Create a random binary map with rectangular forbidden regions.

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

    for _ in range(num_rectangles):
        rect_h = int(torch.randint(min_rect_size, max_rect_size + 1, (1,), device=device).item())
        rect_w = int(torch.randint(min_rect_size, max_rect_size + 1, (1,), device=device).item())

        top = int(torch.randint(1, max(2, map_h - rect_h), (1,), device=device).item())
        left = int(torch.randint(1, max(2, map_w - rect_w), (1,), device=device).item())

        grid[top : top + rect_h, left : left + rect_w] = 0.0

    return grid


def randomize_global_binary_map(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    map_h: int = 8,
    map_w: int = 8,
    map_res: float = 0.5,
    num_rectangles_min: int = 1,
    num_rectangles_max: int = 3,
    min_rect_size: int = 1,
    max_rect_size: int = 2,
    add_border: bool = True,
) -> None:
    """Randomize a global binary map for the selected environments.

    This is intended for use as a reset event.

    Convention:
        0 = forbidden
        1 = allowed
    """
    _ensure_map_metadata(env, map_h, map_w, map_res)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if (not hasattr(env, "plr_global_binary_map")
            or env.plr_global_binary_map.shape != (env.num_envs, map_h, map_w)):
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
