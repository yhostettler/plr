import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from .binary_map_cfg import BinaryMapMarkerCfg

def grid_to_world_xy(rows: torch.Tensor, cols: torch.Tensor, origin_xy: torch.Tensor, resolution: float) -> torch.Tensor:
    x = origin_xy[0] + (cols.float() + 0.5) * resolution
    y = origin_xy[1] + (rows.float() + 0.5) * resolution
    return torch.stack([x, y], dim=-1)


def forbidden_cells_to_world_positions(
    global_map: torch.Tensor,
    origin_xy: torch.Tensor,
    resolution: float,
    z: float = 0.05,
) -> torch.Tensor:
    rows, cols = torch.where(global_map < 0.5)
    if rows.numel() == 0:
        return torch.empty((0, 3), device=global_map.device, dtype=torch.float32)

    xy = grid_to_world_xy(rows, cols, origin_xy, resolution)
    z_col = torch.full((xy.shape[0], 1), z, device=global_map.device, dtype=torch.float32)
    return torch.cat([xy, z_col], dim=-1)


def make_identity_quats(count: int, device: torch.device) -> torch.Tensor:
    quats = torch.zeros((count, 4), device=device, dtype=torch.float32)
    quats[:, 0] = 1.0
    return quats


def create_binary_map_markers():
    robot_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PLRBinaryMap/RobotRoot",
        markers={
            "robot": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )

    sample_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PLRBinaryMap/SampledCells",
        markers={
            "sample": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )

    forbidden_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PLRBinaryMap/ForbiddenCells",
        markers={
            "forbidden": sim_utils.CuboidCfg(
                size=BinaryMapMarkerCfg.FORBIDDEN_CUBE_SIZE,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )

    return {
        "robot": VisualizationMarkers(robot_cfg),
        "sample": VisualizationMarkers(sample_cfg),
        "forbidden": VisualizationMarkers(forbidden_cfg),
    }


def update_robot_marker(marker_dict: dict, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> None:
    marker_dict["robot"].visualize(
        translations=root_pos_w,
        orientations=root_quat_w,
    )


def update_sample_markers(
    marker_dict: dict,
    rows: torch.Tensor,
    cols: torch.Tensor,
    origin_xy: torch.Tensor,
    resolution: float,
    z: float = BinaryMapMarkerCfg.SAMPLE_Z,
) -> torch.Tensor:
    xy = grid_to_world_xy(rows, cols, origin_xy, resolution)
    z_col = torch.full((xy.shape[0], 1), z, device=xy.device, dtype=torch.float32)
    pos = torch.cat([xy, z_col], dim=-1)
    quat = make_identity_quats(pos.shape[0], pos.device)

    marker_dict["sample"].visualize(
        translations=pos,
        orientations=quat,
    )
    return pos


def update_forbidden_markers(
    marker_dict: dict,
    global_map: torch.Tensor,
    origin_xy: torch.Tensor,
    resolution: float,
    z: float = BinaryMapMarkerCfg.FORBIDDEN_Z,
) -> torch.Tensor:
    """Visualize all forbidden cells of the global binary map.

    Args:
        global_map: shape (H, W)
        origin_xy: shape (2,)
        resolution: cell size in meters
        z: z-height for visualization

    Returns:
        Tensor of shape (N, 3) with visualized positions.
    """
    pos = forbidden_cells_to_world_positions(global_map, origin_xy, resolution, z=z)

    # nothing to draw
    if pos.shape[0] == 0:
        print("[markers] no forbidden cells to visualize", flush=True)
        return pos

    quat = make_identity_quats(pos.shape[0], global_map.device)

    marker_dict["forbidden"].visualize(
        translations=pos,
        orientations=quat,
    )
    return pos


