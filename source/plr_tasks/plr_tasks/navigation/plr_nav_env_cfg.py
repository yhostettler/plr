from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import (
    B2WNavigationEnvCfg,
    B2WNavigationEnvCfg_DEV,
    B2WNavigationEnvCfg_PLAY,
)


class PLRB2WNavigationEnvCfg(B2WNavigationEnvCfg):
    """Thin wrapper around the upstream SRU B2W config."""
    pass


class PLRB2WNavigationEnvCfg_DEV(B2WNavigationEnvCfg_DEV):
    """Dev config for quick tests."""
    pass


class PLRB2WNavigationEnvCfg_PLAY(B2WNavigationEnvCfg_PLAY):
    """Play config."""
    pass
