# Currently: 32mx32m rough terrain
# 32x32 with 0.1 resolution
class BinaryMapGeomCfg:
    """Global binary-map geometry."""
    MAP_RES = 0.1
    MAP_H = 320
    MAP_W = 320
    ADD_BORDER = False
    SAFE_CENTER_HALF = 20  # cells kept free around the map center (+-20 cells -> 40x40 safety grid)


class BinaryMapResetCfg:
    """Reset-time randomization of the global map."""
    NUM_RECTANGLES_MIN = 80
    NUM_RECTANGLES_MAX = 100
    MIN_RECT_SIZE = 4
    MAX_RECT_SIZE = 4


class BinaryMapLocalCfg:
    """
    Local egocentric binary map for the teacher.

    LOCAL_SIZE_M / LOCAL_RES = 64, so this gives a 64x64 map
    over a 6.4m x 6.4m local window.
    """
    LOCAL_SIZE_M = 3.2
    LOCAL_RES = 0.2

    LOCAL_H = int(round(LOCAL_SIZE_M / LOCAL_RES))
    LOCAL_W = int(round(LOCAL_SIZE_M / LOCAL_RES))

    # For current Isaac-Lab concatenated observations, keep flattened output.
    # If later you build a CNN teacher path, set this to False.
    FLATTEN_OUTPUT = True

    # Outside the global map should be forbidden.
    OUT_OF_BOUNDS_VALUE = 0.0

    # Keep nearest for exact binary semantics.
    SAMPLE_MODE = "nearest"


class BinaryMapMarkerCfg:
    """Visualization."""
    FORBIDDEN_Z = 0.10
    SAMPLE_Z = 0.15
    FORBIDDEN_CUBE_SIZE = (0.9*BinaryMapGeomCfg.MAP_RES, 0.9*BinaryMapGeomCfg.MAP_RES, 0.06)


class BinaryMapTraceCfg:
    """Training trace."""
    ENABLED = False
    ENV_IDS = [0]
    LIMIT = 50
