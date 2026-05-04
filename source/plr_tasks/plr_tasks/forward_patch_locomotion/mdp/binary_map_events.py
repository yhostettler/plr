import torch
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnv

from .binary_map_cfg import (
    BinaryMapGeomCfg,
    BinaryMapResetCfg,
    BinaryMapCheckerCfg,
    BinaryMapSparseCfg,
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

def _build_soft_penalty_map(binary_map: torch.Tensor, sigma: float = BinaryMapGeomCfg.SOFT_MAP_SIGMA) -> torch.Tensor:
    """Gaussian-blur the forbidden indicator to produce a smooth penalty field.

    Computed once each time the binary map is (re)generated, then cached on env
    as plr_soft_penalty_map.  Per-step cost is a single index lookup.

    Returns (H, W) in [0, 1]: 1.0 at a patch centre, decaying outward.
    Convention matches binary_map: forbidden cells (0.0) become 1.0 in the output.
    """
    radius = int(3 * sigma)
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, device=binary_map.device, dtype=torch.float32)
    kernel_1d = torch.exp(-x ** 2 / (2.0 * sigma ** 2))
    # Unnormalized: peak = 1.0 at distance 0, decaying as exp(-d²/2σ²).
    # Do NOT normalize — we want the map value to reach 1.0 at a patch centre.
    kernel_2d = kernel_1d.outer(kernel_1d).view(1, 1, size, size)

    forbidden = (binary_map < 0.5).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    soft = F.conv2d(forbidden, kernel_2d, padding=radius)
    return soft.squeeze(0).squeeze(0).clamp(0.0, 1.0)  # (H, W)


def place_checkerboard_patches(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    map_h: int = BinaryMapGeomCfg.MAP_H,
    map_w: int = BinaryMapGeomCfg.MAP_W,
    map_res: float = BinaryMapGeomCfg.MAP_RES,
    spawn_clear_m: float = BinaryMapCheckerCfg.SPAWN_CLEAR_M,
    grid_spacing: int = BinaryMapCheckerCfg.GRID_SPACING,
    jitter: int = BinaryMapCheckerCfg.JITTER,
    soft_map_sigma: float = BinaryMapGeomCfg.SOFT_MAP_SIGMA,
) -> None:
    """Place single-cell forbidden patches on a regular grid with random jitter.

    Grid centres are spaced grid_spacing cells apart in both x and y.
    Each centre is displaced by a uniform random offset in [-jitter, +jitter]
    so the layout breaks regularity without forming clusters.

    The leftmost spawn_clear_m metres are kept free so the robot is not
    immediately penalised after spawning.

    Convention: 0 = forbidden, 1 = allowed.
    """
    _ensure_map_metadata(env, map_h, map_w, map_res)

    spawn_clear_cols = int(spawn_clear_m / map_res)
    grid = torch.ones((map_h, map_w), device=env.device, dtype=torch.float32)

    half = grid_spacing // 2
    row_c = half
    while row_c < map_h:
        col_c = spawn_clear_cols + half
        while col_c < map_w:
            rj = int(torch.randint(-jitter, jitter + 1, (1,), device=env.device).item())
            cj = int(torch.randint(-jitter, jitter + 1, (1,), device=env.device).item())
            r = row_c + rj
            c = col_c + cj
            if spawn_clear_cols <= c < map_w and 0 <= r < map_h:
                grid[r, c] = 0.0
            col_c += grid_spacing
        row_c += grid_spacing

    env.plr_global_binary_map = grid
    env.plr_soft_penalty_map = _build_soft_penalty_map(grid, sigma=soft_map_sigma)


# update: modified for new layout (Leon) -> new: single shared global binary map between all environments
# currently not used, changes might be useful later
# def randomize_global_binary_map(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor | None,
#     map_h: int = BinaryMapGeomCfg.MAP_H,
#     map_w: int = BinaryMapGeomCfg.MAP_W,
#     map_res: float = BinaryMapGeomCfg.MAP_RES,
#     num_rectangles_min: int = BinaryMapResetCfg.NUM_RECTANGLES_MIN,
#     num_rectangles_max: int = BinaryMapResetCfg.NUM_RECTANGLES_MAX,
#     min_rect_size: int = BinaryMapResetCfg.MIN_RECT_SIZE,
#     max_rect_size: int = BinaryMapResetCfg.MAX_RECT_SIZE,
#     add_border: bool = BinaryMapGeomCfg.ADD_BORDER,
#     # update: added for new layout (Leon)
#     spawn_clear_m: float = BinaryMapSparseCfg.SPAWN_CLEAR_M,
# ) -> None:
#     """Regenerate the single shared binary map for all environments.

#     The map is a (H, W) tensor shared across all robots — every robot samples
#     its local crop from the same world.  env_ids is accepted for API
#     compatibility but ignored; the whole map is always regenerated.

#     The leftmost spawn_clear_m metres (in x/col) are kept free of patches so
#     robots are never penalised immediately after spawning.

#     Convention: 0 = forbidden, 1 = allowed.
#     """
#     _ensure_map_metadata(env, map_h, map_w, map_res)

#     spawn_clear_cols = int(spawn_clear_m / map_res)

#     num_rectangles = int(
#         torch.randint(num_rectangles_min, num_rectangles_max + 1, (1,), device=env.device).item()
#     )
#     grid = _sample_rectangles_map(
#         map_h=map_h,
#         map_w=map_w,
#         device=env.device,
#         num_rectangles=num_rectangles,
#         min_rect_size=min_rect_size,
#         max_rect_size=max_rect_size,
#         add_border=add_border,
#     )

#     # Clear the spawn zone
#     if spawn_clear_cols > 0:
#         grid[:, :spawn_clear_cols] = 1.0

#     env.plr_global_binary_map = grid




# original function
# def randomize_global_binary_map(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor | None,
#     map_h: int = BinaryMapGeomCfg.MAP_H,
#     map_w: int = BinaryMapGeomCfg.MAP_W,
#     map_res: float = BinaryMapGeomCfg.MAP_RES,
#     num_rectangles_min: int = BinaryMapResetCfg.NUM_RECTANGLES_MIN,
#     num_rectangles_max: int = BinaryMapResetCfg.NUM_RECTANGLES_MAX,
#     min_rect_size: int = BinaryMapResetCfg.MIN_RECT_SIZE,
#     max_rect_size: int = BinaryMapResetCfg.MAX_RECT_SIZE,
#     add_border: bool = BinaryMapGeomCfg.ADD_BORDER,
# ) -> None:
#     """
#     Randomize a global binary map for the selected environments.

#     Convention:
#         0 = forbidden
#         1 = allowed
#     """
#     _ensure_map_metadata(env, map_h, map_w, map_res)

#     if env_ids is None:
#         env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

#     if (
#         not hasattr(env, "plr_global_binary_map")
#         or env.plr_global_binary_map.shape != (env.num_envs, map_h, map_w)
#     ):
#         env.plr_global_binary_map = torch.ones(
#             (env.num_envs, map_h, map_w),
#             device=env.device,
#             dtype=torch.float32,
#         )

#     for env_id in env_ids.tolist():
#         num_rectangles = int(
#             torch.randint(num_rectangles_min, num_rectangles_max + 1, (1,), device=env.device).item()
#         )
#         env.plr_global_binary_map[env_id] = _sample_rectangles_map(
#             map_h=map_h,
#             map_w=map_w,
#             device=env.device,
#             num_rectangles=num_rectangles,
#             min_rect_size=min_rect_size,
#             max_rect_size=max_rect_size,
#             add_border=add_border,
#         )
