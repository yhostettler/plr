import math
import torch
from isaaclab.envs import ManagerBasedRLEnv

from .binary_map_cfg import (
    BinaryMapGeomCfg,
    BinaryMapHumanCfg,
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


def _ensure_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    """Return valid env ids tensor on the correct device."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    return env_ids


def _world_to_grid(
    xy_world: torch.Tensor,
    map_origin_xy: torch.Tensor,
    map_res: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert world xy positions [*, 2] to integer map indices.

    Returns:
        row, col tensors with same leading shape.
    """
    col = torch.floor((xy_world[..., 0] - map_origin_xy[0]) / map_res).long()
    row = torch.floor((xy_world[..., 1] - map_origin_xy[1]) / map_res).long()
    return row, col


def _stamp_rect_footprints(
    footprint_map: torch.Tensor,
    foot_centers_xy: torch.Tensor,   # [M, 4] = [env_id, x, y, yaw]
    map_origin_xy: torch.Tensor,
    map_res: float,
    foot_length_m: float,
    foot_width_m: float,
    footprint_value: float,
) -> None:
    """Stamp yaw-rotated rectangular footprints into footprint_map."""
    if foot_centers_xy.numel() == 0:
        return

    _, map_h, map_w = footprint_map.shape

    foot_len_cells = max(1, int(round(foot_length_m / map_res)))
    foot_wid_cells = max(1, int(round(foot_width_m / map_res)))

    half_len = foot_len_cells / 2.0
    half_wid = foot_wid_cells / 2.0

    env_ids = foot_centers_xy[:, 0].long()
    xy = foot_centers_xy[:, 1:3]
    yaws = foot_centers_xy[:, 3]

    rows, cols = _world_to_grid(xy, map_origin_xy, map_res)

    # search window slightly larger than footprint bbox
    radius = int(math.ceil(max(foot_len_cells, foot_wid_cells))) + 1

    for i in range(env_ids.shape[0]):
        env_id = int(env_ids[i].item())
        row_c = int(rows[i].item())
        col_c = int(cols[i].item())
        yaw = float(yaws[i].item())

        r0 = max(0, row_c - radius)
        r1 = min(map_h, row_c + radius + 1)
        c0 = max(0, col_c - radius)
        c1 = min(map_w, col_c + radius + 1)

        if r0 >= r1 or c0 >= c1:
            continue

        rr = torch.arange(r0, r1, device=footprint_map.device, dtype=torch.float32)
        cc = torch.arange(c0, c1, device=footprint_map.device, dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(rr, cc, indexing="ij")

        # local coordinates in cell units relative to footprint center
        dx = grid_c - float(col_c)   # forward-ish axis before rotation
        dy = grid_r - float(row_c)

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # rotate grid into footprint local frame
        local_x =  cos_y * dx + sin_y * dy
        local_y = -sin_y * dx + cos_y * dy

        mask = (local_x.abs() <= half_len) & (local_y.abs() <= half_wid)
        footprint_map[env_id, r0:r1, c0:c1][mask] = footprint_value



def _sample_spawn_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_humans: int,
    map_h: int,
    map_w: int,
    map_res: float,
    spawn_margin_m: float,
) -> torch.Tensor:
    """
    Sample initial human positions inside the map margins.

    Returns:
        pos_xy: [len(env_ids), num_humans, 2]
    """
    num_sel = env_ids.shape[0]
    map_origin = env.plr_map_origin_xy
    x_min = map_origin[0] + spawn_margin_m
    x_max = map_origin[0] + map_w * map_res - spawn_margin_m
    y_min = map_origin[1] + spawn_margin_m
    y_max = map_origin[1] + map_h * map_res - spawn_margin_m

    pos_xy = torch.empty((num_sel, num_humans, 2), device=env.device, dtype=torch.float32)
    pos_xy[..., 0] = torch.rand((num_sel, num_humans), device=env.device) * (x_max - x_min) + x_min
    pos_xy[..., 1] = torch.rand((num_sel, num_humans), device=env.device) * (y_max - y_min) + y_min
    return pos_xy


def _sample_spawn_yaws(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_humans: int,
    yaw_range: tuple[float, float],
) -> torch.Tensor:
    """Sample initial human headings."""
    yaw_min, yaw_max = yaw_range
    num_sel = env_ids.shape[0]
    return torch.rand((num_sel, num_humans), device=env.device) * (yaw_max - yaw_min) + yaw_min


def _place_current_feet(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    step_width_m: float,
    foot_length_m: float,
    foot_width_m: float,
    footprint_value: float,
) -> None:
    """
    Stamp one footprint per walker using the current phase.

    phase == 0 -> next foot is left
    phase == 1 -> next foot is right
    """
    pos_xy = env.plr_human_pos_xy[env_ids]         # [E, K, 2]
    yaw = env.plr_human_yaw[env_ids]               # [E, K]
    phase = env.plr_human_phase[env_ids]           # [E, K]

    left_lat = step_width_m / 2.0
    right_lat = -step_width_m / 2.0
    lateral_offset = torch.where(
        phase == 0,
        torch.full_like(yaw, left_lat),
        torch.full_like(yaw, right_lat),
    )

    lateral_dir = torch.stack(
        (-torch.sin(yaw), torch.cos(yaw)),
        dim=-1,
    )  # [E, K, 2]

    foot_xy = pos_xy + lateral_offset.unsqueeze(-1) * lateral_dir

    e_idx, h_idx = torch.meshgrid(
        torch.arange(env_ids.shape[0], device=env.device),
        torch.arange(pos_xy.shape[1], device=env.device),
        indexing="ij",
    )

    stamp_data = torch.stack(
        (
            env_ids[e_idx.reshape(-1)].float(),
            foot_xy[..., 0].reshape(-1),
            foot_xy[..., 1].reshape(-1),
            yaw.reshape(-1),
        ),
        dim=-1,
    )

    _stamp_rect_footprints(
        footprint_map=env.plr_human_footprint_map,
        foot_centers_xy=stamp_data,
        map_origin_xy=env.plr_map_origin_xy,
        map_res=env.plr_map_resolution,
        foot_length_m=foot_length_m,
        foot_width_m=foot_width_m,
        footprint_value=footprint_value,
    )


def _wrap_walkers_in_map(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    map_h: int,
    map_w: int,
    map_res: float,
    spawn_margin_m: float,
) -> None:
    """
    Wrap walkers to the opposite side when they leave the valid spawn region.
    """
    pos_xy = env.plr_human_pos_xy[env_ids]
    yaw = env.plr_human_yaw[env_ids]

    origin = env.plr_map_origin_xy
    x_min = origin[0] + spawn_margin_m
    x_max = origin[0] + map_w * map_res - spawn_margin_m
    y_min = origin[1] + spawn_margin_m
    y_max = origin[1] + map_h * map_res - spawn_margin_m

    out_left = pos_xy[..., 0] < x_min
    out_right = pos_xy[..., 0] > x_max
    out_bottom = pos_xy[..., 1] < y_min
    out_top = pos_xy[..., 1] > y_max

    pos_xy[..., 0] = torch.where(out_left, torch.full_like(pos_xy[..., 0], x_max), pos_xy[..., 0])
    pos_xy[..., 0] = torch.where(out_right, torch.full_like(pos_xy[..., 0], x_min), pos_xy[..., 0])
    pos_xy[..., 1] = torch.where(out_bottom, torch.full_like(pos_xy[..., 1], y_max), pos_xy[..., 1])
    pos_xy[..., 1] = torch.where(out_top, torch.full_like(pos_xy[..., 1], y_min), pos_xy[..., 1])

    env.plr_human_pos_xy[env_ids] = pos_xy
    env.plr_human_yaw[env_ids] = yaw


def reset_human_footstep_map(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    map_h: int = BinaryMapGeomCfg.MAP_H,
    map_w: int = BinaryMapGeomCfg.MAP_W,
    map_res: float = BinaryMapGeomCfg.MAP_RES,
    num_humans: int = BinaryMapHumanCfg.NUM_HUMANS,
    default_speed_mps: float = BinaryMapHumanCfg.DEFAULT_SPEED_MPS,
    default_yaw_range: tuple[float, float] = BinaryMapHumanCfg.DEFAULT_YAW_RANGE,
    step_period_s: float = BinaryMapHumanCfg.STEP_PERIOD_S,
    stride_length_m: float = BinaryMapHumanCfg.STRIDE_LENGTH_M,
    step_width_m: float = BinaryMapHumanCfg.STEP_WIDTH_M,
    foot_length_m: float = BinaryMapHumanCfg.FOOT_LENGTH_M,
    foot_width_m: float = BinaryMapHumanCfg.FOOT_WIDTH_M,
    spawn_margin_m: float = BinaryMapHumanCfg.SPAWN_MARGIN_M,
    max_footprint_age: int = BinaryMapHumanCfg.MAX_FOOTPRINT_AGE,
    footprint_value: float = BinaryMapHumanCfg.FOOTPRINT_VALUE,
    background_value: float = BinaryMapHumanCfg.BACKGROUND_VALUE,
) -> None:
    """
    Initialize deterministic human walkers and create the first global footprint map.

    Convention is inherited from cfg:
        footprint_value = allowed or forbidden, depending on your setup
    """
    del stride_length_m  # not needed directly in reset, but kept in signature for config symmetry

    _ensure_map_metadata(env, map_h, map_w, map_res)
    env_ids = _ensure_env_ids(env, env_ids)

    if not hasattr(env, "plr_global_binary_map") or env.plr_global_binary_map.shape != (env.num_envs, map_h, map_w):
        env.plr_global_binary_map = torch.full(
            (env.num_envs, map_h, map_w),
            background_value,
            device=env.device,
            dtype=torch.float32,
        )

    if not hasattr(env, "plr_human_footprint_map") or env.plr_human_footprint_map.shape != (env.num_envs, map_h, map_w):
        env.plr_human_footprint_map = torch.full(
            (env.num_envs, map_h, map_w),
            background_value,
            device=env.device,
            dtype=torch.float32,
        )

    if not hasattr(env, "plr_human_footprint_age") or env.plr_human_footprint_age.shape != (env.num_envs, map_h, map_w):
        env.plr_human_footprint_age = torch.zeros(
            (env.num_envs, map_h, map_w),
            device=env.device,
            dtype=torch.int32,
        )

    if not hasattr(env, "plr_human_pos_xy") or env.plr_human_pos_xy.shape != (env.num_envs, num_humans, 2):
        env.plr_human_pos_xy = torch.zeros(
            (env.num_envs, num_humans, 2), device=env.device, dtype=torch.float32
        )
        env.plr_human_yaw = torch.zeros(
            (env.num_envs, num_humans), device=env.device, dtype=torch.float32
        )
        env.plr_human_speed = torch.zeros(
            (env.num_envs, num_humans), device=env.device, dtype=torch.float32
        )
        env.plr_human_phase = torch.zeros(
            (env.num_envs, num_humans), device=env.device, dtype=torch.long
        )
        env.plr_human_step_timer = torch.zeros(
            (env.num_envs, num_humans), device=env.device, dtype=torch.float32
        )

    env.plr_human_footprint_map[env_ids] = background_value
    env.plr_human_footprint_age[env_ids] = 0
    env.plr_global_binary_map[env_ids] = background_value

    env.plr_human_pos_xy[env_ids] = _sample_spawn_positions(
        env=env,
        env_ids=env_ids,
        num_humans=num_humans,
        map_h=map_h,
        map_w=map_w,
        map_res=map_res,
        spawn_margin_m=spawn_margin_m,
    )
    env.plr_human_yaw[env_ids] = _sample_spawn_yaws(
        env=env,
        env_ids=env_ids,
        num_humans=num_humans,
        yaw_range=default_yaw_range,
    )
    env.plr_human_speed[env_ids] = default_speed_mps

    # 0 = left next, 1 = right next
    env.plr_human_phase[env_ids] = 0

    # Start at random offset inside the step cycle so walkers do not stamp in perfect sync
    env.plr_human_step_timer[env_ids] = torch.rand(
        (env_ids.shape[0], num_humans), device=env.device, dtype=torch.float32
    ) * step_period_s

    # Stamp initial left foot
    _place_current_feet(
        env=env,
        env_ids=env_ids,
        step_width_m=step_width_m,
        foot_length_m=foot_length_m,
        foot_width_m=foot_width_m,
        footprint_value=footprint_value,
    )

    # Age map for just-stamped cells
    stamped_mask = env.plr_human_footprint_map[env_ids] == footprint_value
    env.plr_human_footprint_age[env_ids][stamped_mask] = max_footprint_age

    # Toggle and stamp the opposite foot once so each walker starts with two visible feet
    env.plr_human_phase[env_ids] = 1
    _place_current_feet(
        env=env,
        env_ids=env_ids,
        step_width_m=step_width_m,
        foot_length_m=foot_length_m,
        foot_width_m=foot_width_m,
        footprint_value=footprint_value,
    )

    stamped_mask = env.plr_human_footprint_map[env_ids] == footprint_value
    env.plr_human_footprint_age[env_ids][stamped_mask] = max_footprint_age

    # Next foot after reset is left again
    env.plr_human_phase[env_ids] = 0

    # First version: global map is only the footprint map
    env.plr_global_binary_map[env_ids] = env.plr_human_footprint_map[env_ids]


def update_human_footstep_map(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    dt: float,
    map_h: int = BinaryMapGeomCfg.MAP_H,
    map_w: int = BinaryMapGeomCfg.MAP_W,
    map_res: float = BinaryMapGeomCfg.MAP_RES,
    num_humans: int = BinaryMapHumanCfg.NUM_HUMANS,
    step_period_s: float = BinaryMapHumanCfg.STEP_PERIOD_S,
    stride_length_m: float = BinaryMapHumanCfg.STRIDE_LENGTH_M,
    step_width_m: float = BinaryMapHumanCfg.STEP_WIDTH_M,
    foot_length_m: float = BinaryMapHumanCfg.FOOT_LENGTH_M,
    foot_width_m: float = BinaryMapHumanCfg.FOOT_WIDTH_M,
    spawn_margin_m: float = BinaryMapHumanCfg.SPAWN_MARGIN_M,
    max_footprint_age: int = BinaryMapHumanCfg.MAX_FOOTPRINT_AGE,
    footprint_value: float = BinaryMapHumanCfg.FOOTPRINT_VALUE,
    background_value: float = BinaryMapHumanCfg.BACKGROUND_VALUE,
) -> None:
    """
    Advance walkers and stamp footprints when their step timer fires.
    """
    del num_humans  # shape already fixed by reset

    env_ids = _ensure_env_ids(env, env_ids)

    if not hasattr(env, "plr_human_pos_xy"):
        raise RuntimeError(
            "Human footstep map state not initialized. "
            "Make sure reset_human_footstep_map is called as a reset event first."
        )

    # 1) Move walkers continuously
    yaw = env.plr_human_yaw[env_ids]
    speed = env.plr_human_speed[env_ids]
    forward_dir = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)

    env.plr_human_pos_xy[env_ids] = env.plr_human_pos_xy[env_ids] + forward_dir * speed.unsqueeze(-1) * dt

    # 2) Keep walkers inside map by wrapping around
    _wrap_walkers_in_map(
        env=env,
        env_ids=env_ids,
        map_h=map_h,
        map_w=map_w,
        map_res=map_res,
        spawn_margin_m=spawn_margin_m,
    )

    # 3) Age out old footprints
    if BinaryMapHumanCfg.AGING_FOOTSTEPS:
        age = env.plr_human_footprint_age[env_ids]
        age = torch.clamp(age - 1, min=0)
        env.plr_human_footprint_age[env_ids] = age

        env.plr_human_footprint_map[env_ids] = torch.where(
            age > 0,
            torch.full_like(env.plr_human_footprint_map[env_ids], footprint_value),
            torch.full_like(env.plr_human_footprint_map[env_ids], background_value),
        )

    # 4) Advance step timers
    env.plr_human_step_timer[env_ids] = env.plr_human_step_timer[env_ids] + dt

    step_mask = env.plr_human_step_timer[env_ids] >= step_period_s
    if torch.any(step_mask):
        # Move body forward by one stride at the instant of stepping.
        # This keeps footprint spacing close to stride_length.
        step_forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1) * stride_length_m
        env.plr_human_pos_xy[env_ids] = torch.where(
            step_mask.unsqueeze(-1),
            env.plr_human_pos_xy[env_ids] + step_forward,
            env.plr_human_pos_xy[env_ids],
        )

        _wrap_walkers_in_map(
            env=env,
            env_ids=env_ids,
            map_h=map_h,
            map_w=map_w,
            map_res=map_res,
            spawn_margin_m=spawn_margin_m,
        )

        # Stamp the current next foot for walkers whose timer fired
        pos_xy = env.plr_human_pos_xy[env_ids]
        phase = env.plr_human_phase[env_ids]

        lateral_offset = torch.where(
            phase == 0,
            torch.full_like(yaw, step_width_m / 2.0),
            torch.full_like(yaw, -step_width_m / 2.0),
        )
        lateral_dir = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        foot_xy = pos_xy + lateral_offset.unsqueeze(-1) * lateral_dir

        sel_env_local, sel_human = torch.where(step_mask)
        stamp_data = torch.stack(
            (
                env_ids[sel_env_local].float(),
                foot_xy[sel_env_local, sel_human, 0],
                foot_xy[sel_env_local, sel_human, 1],
                yaw[sel_env_local, sel_human],
            ),
            dim=-1,
        )

        _stamp_rect_footprints(
            footprint_map=env.plr_human_footprint_map,
            foot_centers_xy=stamp_data,
            map_origin_xy=env.plr_map_origin_xy,
            map_res=env.plr_map_resolution,
            foot_length_m=foot_length_m,
            foot_width_m=foot_width_m,
            footprint_value=footprint_value,
        )

        # Reset age only where just stamped
        rows, cols = _world_to_grid(foot_xy[sel_env_local, sel_human], env.plr_map_origin_xy, env.plr_map_resolution)
        foot_len_cells = max(1, int(round(foot_length_m / env.plr_map_resolution)))
        foot_wid_cells = max(1, int(round(foot_width_m / env.plr_map_resolution)))
        half_h = foot_wid_cells // 2
        half_w = foot_len_cells // 2

        for i in range(sel_env_local.shape[0]):
            env_id = int(env_ids[sel_env_local[i]].item())
            row = int(rows[i].item())
            col = int(cols[i].item())

            r0 = max(0, row - half_h)
            r1 = min(map_h, row + half_h + 1)
            c0 = max(0, col - half_w)
            c1 = min(map_w, col + half_w + 1)

            if r0 < r1 and c0 < c1:
                env.plr_human_footprint_age[env_id, r0:r1, c0:c1] = max_footprint_age

        # Toggle left/right phase only for walkers that stepped
        env.plr_human_phase[env_ids] = torch.where(
            step_mask,
            1 - env.plr_human_phase[env_ids],
            env.plr_human_phase[env_ids],
        )

        # Keep residual timer instead of setting exactly to zero
        env.plr_human_step_timer[env_ids] = torch.where(
            step_mask,
            env.plr_human_step_timer[env_ids] - step_period_s,
            env.plr_human_step_timer[env_ids],
        )

    # 5) First version: global map is just the human footprint map
    env.plr_global_binary_map[env_ids] = env.plr_human_footprint_map[env_ids]
