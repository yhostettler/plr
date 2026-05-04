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
    safe_center_half: int = BinaryMapGeomCfg.SAFE_CENTER_HALF,
) -> torch.Tensor:
    """
    Create a random binary map with rectangular forbidden regions.

    Convention:
        0 = forbidden
        1 = allowed

    A square of side 2*safe_center_half centered on the map is always kept free,
    regardless of where rectangles are placed.
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

    # Always restore the central spawn zone — robots always start here. Tipp: set to 0.0 to visually inspect area as forbidden zone
    cy, cx = map_h // 2, map_w // 2
    grid[cy - safe_center_half : cy + safe_center_half, cx - safe_center_half : cx + safe_center_half] = 1.0

    return grid


def _chessboard_map(
    map_h: int,
    map_w: int,
    device: torch.device,
    patch_size: int,
    stride: int,
    add_border: bool,
    jitter: int = BinaryMapResetCfg.CHESSBOARD_JITTER,
    safe_center_half: int = BinaryMapGeomCfg.SAFE_CENTER_HALF,
) -> torch.Tensor:
    """Create a chessboard-style binary map with optional per-patch position jitter.

    Patches are placed on every other grid position (where (i + j) % 2 == 0).
    Each patch center is perturbed by a random offset in [-jitter, +jitter] cells,
    so the edge-to-edge gap varies by ±2*jitter cells around the nominal value.

    Convention: 0 = forbidden, 1 = allowed.
    """
    grid = torch.ones((map_h, map_w), device=device, dtype=torch.float32)

    if add_border:
        grid[0, :] = 0.0
        grid[-1, :] = 0.0
        grid[:, 0] = 0.0
        grid[:, -1] = 0.0

    n_rows = map_h // stride
    n_cols = map_w // stride

    # Center the grid in the map so the pattern is symmetric.
    row_offset = (map_h - n_rows * stride) // 2
    col_offset = (map_w - n_cols * stride) // 2

    half = patch_size // 2

    for i in range(n_rows):
        for j in range(n_cols):
            if (i + j) % 2 != 0:  # chessboard: skip every other position
                continue

            # Nominal patch center for this grid cell.
            cy = row_offset + i * stride + stride // 2
            cx = col_offset + j * stride + stride // 2

            # Random per-patch jitter in [-jitter, +jitter].
            if jitter > 0:
                cy += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())
                cx += int(torch.randint(-jitter, jitter + 1, (1,), device=device).item())

            r0 = max(0, cy - half)
            r1 = min(map_h, cy - half + patch_size)
            c0 = max(0, cx - half)
            c1 = min(map_w, cx - half + patch_size)
            grid[r0:r1, c0:c1] = 0.0

    # Always restore the central spawn zone.
    cy_map, cx_map = map_h // 2, map_w // 2
    grid[cy_map - safe_center_half : cy_map + safe_center_half,
         cx_map - safe_center_half : cx_map + safe_center_half] = 1.0

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
    use_chessboard: bool = BinaryMapResetCfg.USE_CHESSBOARD,
    chessboard_stride: int = BinaryMapResetCfg.CHESSBOARD_STRIDE,
    chessboard_jitter: int = BinaryMapResetCfg.CHESSBOARD_JITTER,
) -> None:
    """Regenerate the single shared binary map for all environments.

    The map is a (H, W) tensor shared across all robots — every robot samples
    its local crop from the same world. env_ids is accepted for API
    compatibility but ignored; the whole map is always regenerated.

    Convention:
        0 = forbidden
        1 = allowed
    """
    _ensure_map_metadata(env, map_h, map_w, map_res)

    if use_chessboard:
        env.plr_global_binary_map = _chessboard_map(
            map_h=map_h,
            map_w=map_w,
            device=env.device,
            patch_size=min_rect_size,
            stride=chessboard_stride,
            add_border=add_border,
            jitter=chessboard_jitter,
        )
    else:
        num_rectangles = int(
            torch.randint(num_rectangles_min, num_rectangles_max + 1, (1,), device=env.device).item()
        )
        env.plr_global_binary_map = _sample_rectangles_map(
            map_h=map_h,
            map_w=map_w,
            device=env.device,
            num_rectangles=num_rectangles,
            min_rect_size=min_rect_size,
            max_rect_size=max_rect_size,
            add_border=add_border,
        )
