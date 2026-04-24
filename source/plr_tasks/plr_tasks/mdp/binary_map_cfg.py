class BinaryMapGeomCfg:
    """Global binary-map geometry."""
    MAP_RES = 0.1
    MAP_H = 200
    MAP_W = 200
    ADD_BORDER = False


class BinaryMapResetCfg:
    """Reset-time randomization of the global map."""
    NUM_RECTANGLES_MIN = 25
    NUM_RECTANGLES_MAX = 40
    MIN_RECT_SIZE = 1
    MAX_RECT_SIZE = 5


class BinaryMapIntervalCfg:
    """Dynamic updates during the episode."""
    ENABLED = True
    INTERVAL_RANGE_S = (1.0, 1.0)
    NUM_PATCHES = 10
    PATCH_SIZE = 5


class BinaryMapLocalCfg:
    """
    Local egocentric binary map for the teacher.

    LOCAL_SIZE_M / LOCAL_RES = 64, so this gives a 64x64 map
    over a 6.4m x 6.4m local window.
    """
    LOCAL_SIZE_M = 1.6
    LOCAL_RES = 0.1

    LOCAL_H = int(round(LOCAL_SIZE_M / LOCAL_RES))
    LOCAL_W = int(round(LOCAL_SIZE_M / LOCAL_RES))

    # For current Isaac-Lab concatenated observations, keep flattened output.
    # If later you build a CNN teacher path, set this to False.
    FLATTEN_OUTPUT = True

    # Outside the global map should be forbidden.
    OUT_OF_BOUNDS_VALUE = 0.0

    # Keep nearest for exact binary semantics.
    SAMPLE_MODE = "nearest"

    # Helpful safety checks
    assert LOCAL_H == 16
    assert LOCAL_W == 16


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
    FORBIDDEN_Z = 0.10
    SAMPLE_Z = 0.15
    FORBIDDEN_CUBE_SIZE = (0.10, 0.10, 0.06)


class BinaryMapTraceCfg:
    """Training trace."""
    ENABLED = False
    ENV_IDS = [0]
    LIMIT = 50
