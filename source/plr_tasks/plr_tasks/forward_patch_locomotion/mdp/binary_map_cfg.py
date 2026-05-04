# updat: changed for new layout (Leon)
class BinaryMapGeomCfg:
    """Global binary-map geometry.

    At MAP_RES=0.5 m/cell the 80×80 grid covers a 40m×40m square,
    matching the 5-column × 5-row terrain (5 × 8m = 40m per side).
    """
    MAP_RES = 0.5
    MAP_H = 80   # cells in y (lateral)  — 80 × 0.5 m = 40 m
    MAP_W = 80   # cells in x (forward)  — 80 × 0.5 m = 40 m
    ADD_BORDER = False
    # Gaussian sigma for the soft penalty map (in map cells).
    # Rule of thumb: penalty drops to ~5 % at distance d_max_m from a patch:
    #   SOFT_MAP_SIGMA = d_max_m / MAP_RES / sqrt(2 * ln(20))
    # 1.2 cells × 0.5 m/cell = 0.6 m sigma  →  ~5 % cutoff at 1.5 m
    SOFT_MAP_SIGMA: float = 1.2

# updat: changed for new layout (Leon)
class BinaryMapResetCfg:
    """Reset-time randomization of the global map (uniform random placement)."""
    NUM_RECTANGLES_MIN = 80
    NUM_RECTANGLES_MAX = 120
    MIN_RECT_SIZE = 1   # cells — 0.5 m at 0.5 m/cell
    MAX_RECT_SIZE = 2   # cells — 1 m

# updat: added for new layout (Leon)
class BinaryMapCheckerCfg:
    """Quasi-checkerboard patch placement.

    Patches are placed on a regular grid with small random jitter so the
    layout looks evenly scattered (no clusters) but not perfectly mechanical.
    Each forbidden patch is a single cell (MAP_RES × MAP_RES = 0.5 m × 0.5 m).

    At MAP_RES=0.5 m/cell, GRID_SPACING=7 cells = 3.5 m between centers.
    Active area after spawn clear: 64 cols × 80 rows → ~9×11 = ~99 patches.

    SPAWN_CLEAR_M = 8 m clears 16 cols from map left edge.  Robots spawn at
    col ≈ 8 (x ≈ −16 m), so the first 4 m ahead of spawn is always free.
    """
    SPAWN_CLEAR_M = 8.0
    GRID_SPACING = 7   # cells between patch centres
    JITTER = 2         # ±cells random offset per patch

# updat: added for new layout (Leon)
class BinaryMapSparseCfg:
    """Sparse isolated patches spread along the robot's forward (x) axis.

    Layout: a clear zone at the left of the map (near spawn), then one small
    patch at a time placed at a random y, separated by variable gaps along x.

    Patches are elongated in x (direction of travel) and thin in y so they
    resemble a footprint-sized obstacle the robot must actively avoid.

    At MAP_RES=0.5 m/cell:
        SPAWN_CLEAR_M = 5 m  → 10 cells left edge kept free
        MIN_SPACING_M = 2 m  → at least 4 cells between patches
        MAX_SPACING_M = 5 m  → at most 10 cells between patches
        patch y-size: 1 cell = 0.5 m  (thin lateral footprint)
        patch x-size: 2–3 cells = 1–1.5 m (elongated along movement)
    → expect ~8–12 patches per map across the 35 m active corridor
    """
    SPAWN_CLEAR_M = 5.0
    MIN_SPACING_M = 0.5
    MAX_SPACING_M = 1.5
    PATCH_H_MIN = 1
    PATCH_H_MAX = 1
    PATCH_W_MIN = 2
    PATCH_W_MAX = 3



class BinaryMapIntervalCfg:
    """Dynamic updates during the episode."""
    ENABLED = True
    INTERVAL_RANGE_S = (1.0, 1.0)
    NUM_PATCHES = 10
    PATCH_SIZE = 5


class BinaryMapLocalCfg:
    """Local egocentric binary map crop fed as MLP observation.

    12×12 cells at 0.5 m/cell → 6 m × 6 m window around the robot.
    Flattened to 144 values.
    """
    LOCAL_SIZE_M = 6.0
    LOCAL_RES = 0.5

    LOCAL_H = int(round(LOCAL_SIZE_M / LOCAL_RES))   # 12
    LOCAL_W = int(round(LOCAL_SIZE_M / LOCAL_RES))   # 12

    FLATTEN_OUTPUT = True
    OUT_OF_BOUNDS_VALUE = 0.0
    SAMPLE_MODE = "nearest"

    assert LOCAL_H == 12
    assert LOCAL_W == 12


class BinaryMapHumanCfg:
    ENABLED = True

    # number of walking humans per env
    NUM_HUMANS = 3

    # fixed default motion for first version
    DEFAULT_SPEED_MPS = 1.2
    DEFAULT_YAW_RANGE = (-3.14159, 3.14159)

    # gait / footprint timing
    STEP_PERIOD_S = 0.5
    STRIDE_LENGTH_M = 0.6
    STEP_WIDTH_M = 0.18

    # footprint size
    FOOT_LENGTH_M = 0.26
    FOOT_WIDTH_M = 0.10

    # where walkers spawn
    SPAWN_MARGIN_M = 1.0

    # how many recent steps stay visible
    MAX_FOOTPRINT_AGE = 20

    # if 1 = allowed patch
    # if 0 = forbidden patch
    FOOTPRINT_VALUE = 0.0
    BACKGROUND_VALUE = 1.0

class BinaryMapMarkerCfg:
    """Visualization."""
    FORBIDDEN_Z = 0.05
    SAMPLE_Z = 0.15
    FORBIDDEN_CUBE_SIZE = (0.5, 0.5, 0.06)  # updat: changed for new layout (Leon)


class BinaryMapTraceCfg:
    """Training trace."""
    ENABLED = False
    ENV_IDS = [0]
    LIMIT = 50
