from isaaclab_nav_task.navigation.config.b2w.agents.rsl_rl_cfg import (
    B2WNavPPORunnerCfg,
    B2WNavPPORunnerDevCfg,
)


class PLRB2WNavPPORunnerCfg(B2WNavPPORunnerCfg):
    """Thin wrapper around the upstream PPO runner config."""
    pass


class PLRB2WNavPPORunnerDevCfg(B2WNavPPORunnerDevCfg):
    """Thin wrapper around the upstream PPO dev runner config."""
    pass
