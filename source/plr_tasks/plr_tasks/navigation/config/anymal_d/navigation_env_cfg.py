# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""AoW-D (Anymal on Wheels) specific configuration for navigation environment."""

from isaaclab.utils import configclass

from plr_tasks.navigation.navigation_env_cfg import NavigationEnvCfg
import plr_tasks.navigation.mdp as mdp

#from plr_tasks.navigation.assets import ANYMAL_D_ON_WHEELS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip

@configclass
class ANYmalDNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # from plr_tasks.navigation.mdp.observations import initialize_depth_noise_generator
        # from plr_tasks.navigation.mdp.depth_utils.camera_config import get_camera_config

        # initialize_depth_noise_generator(robot_name="aow_d", use_jit_precompiled=False)

        # camera_config = get_camera_config("aow_d")
        # CAMERA_RESOLUTION = camera_config.resolution

        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # self.scene.terrain.max_init_terrain_level = 10
        # self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        # self.scene.terrain.terrain_generator.curriculum = False


        # added
        # Simplify rewards for velocity tracking  
        # self.rewards.goal_reaching.weight = 0.0  # Disable goal reaching  
        # self.rewards.velocity_tracking.weight = 1.0  # Enable velocity tracking  
          
        # Remove curriculum learning  
        # self.curriculum.terrain_levels.enabled = False

# @configclass
# class AowDNavigationEnvCfg_DEV(AowDNavigationEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.terrain.terrain_generator.num_rows = 1
#         self.scene.terrain.terrain_generator.num_cols = 30
#         self.scene.terrain.max_init_terrain_level = 10
#         self.scene.terrain.terrain_generator.difficulty_range = [0.4, 0.8]
#         self.scene.terrain.terrain_generator.curriculum = True
#         self.scene.num_envs = 1

@configclass
class ANYmalDNavigationEnvCfg_PLAY(ANYmalDNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
