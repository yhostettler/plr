from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from plr_tasks.navigation import mdp as plr_mdp

from isaaclab_nav_task.navigation.navigation_env_cfg import ObservationsCfg as SRUObservationsCfg
from isaaclab_nav_task.navigation.config.b2w.navigation_env_cfg import (
    B2WNavigationEnvCfg,
    B2WNavigationEnvCfg_DEV,
    B2WNavigationEnvCfg_PLAY,
)

@configclass
class PLRObservationsCfg(SRUObservationsCfg):

    @configclass
    class PolicyCfg(SRUObservationsCfg.PolicyCfg):
        binary_map_2x2 = ObsTerm(func=plr_mdp.fixed_binary_2x2)

        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = True

    @configclass
    class CriticCfg(SRUObservationsCfg.CriticCfg):
        binary_map_2x2 = ObsTerm(func=plr_mdp.fixed_binary_2x2)

        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class PLRB2WNavigationEnvCfg(B2WNavigationEnvCfg):
    observations: PLRObservationsCfg = PLRObservationsCfg()


@configclass
class PLRB2WNavigationEnvCfg_DEV(B2WNavigationEnvCfg_DEV):
    observations: PLRObservationsCfg = PLRObservationsCfg()


@configclass
class PLRB2WNavigationEnvCfg_PLAY(B2WNavigationEnvCfg_PLAY):
    observations: PLRObservationsCfg = PLRObservationsCfg()

