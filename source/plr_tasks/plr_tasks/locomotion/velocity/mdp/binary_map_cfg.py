class BinaryMapGeomCfg:
    # global map geometry (MAP_H * MAP_RES in meters)
    MAP_H = 64
    MAP_W = MAP_H
    MAP_RES = 0.2
    ADD_BORDER = False


class BinaryMapResetCfg:
    # reset randomization
    NUM_RECTANGLES_MIN = 10
    NUM_RECTANGLES_MAX = 15
    MIN_RECT_SIZE = 1
    MAX_RECT_SIZE = 5


class BinaryMapIntervalCfg:
    # dynamic updates during the episode
    ENABLED = True
    INTERVAL_RANGE_S = (1.0, 1.0)
    NUM_PATCHES = 5
    PATCH_SIZE = 1


class BinaryMapEgoCfg:
    # local 2x2 sampling offsets in robot frame
    EGO_HALF_SPAN_X = 0.15
    EGO_HALF_SPAN_Y = 0.15


class BinaryMapMarkerCfg:
    # visualization
    FORBIDDEN_Z = 0.10
    SAMPLE_Z = 0.15
    FORBIDDEN_CUBE_SIZE = (0.18, 0.18, 0.06)    #Single source of truth for global binary map settings.

class BinaryMapTraceCfg:
    #Training Trace
    ENABLED = True
    ENV_IDS = [0]
    LIMIT = 50
