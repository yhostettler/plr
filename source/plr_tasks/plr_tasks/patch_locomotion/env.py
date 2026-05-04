# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom navigation environment with observation delay support."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv

from plr_tasks.patch_locomotion.managers.ema_manager import EMAManager
import plr_tasks.patch_locomotion.mdp.markers as mdp_markers

if TYPE_CHECKING:
    from plr_tasks.patch_locomotion.env_cfg import PLRPatchLocomotionEnvCfg


class PLRLocomotionEnv(ManagerBasedRLEnv):
    """Navigation environment with EMA manager and optional binary map visualization.

    Extends ManagerBasedRLEnv with:
    - EMAManager created before other managers so observations can reference it.
    - Binary map debug visualization: set cfg.viewer.debug_vis = True to draw
      forbidden cells of env 0 as red cubes in the Isaac Sim viewport.

    Note: ManagerBasedRLEnv has no set_debug_vis hook (unlike DirectRLEnv), so
    the debug_vis flag is read once in load_managers and markers are created there.
    Per-step updates happen in the step() override, which runner.learn() calls
    on every training step — that is the only injection point available.
    """

    cfg: PLRPatchLocomotionEnvCfg

    def load_managers(self):
        # Create the EMA manager before other managers so observation terms
        # can reference env.ema_manager during their own __init__.
        self.ema_manager = EMAManager(
            cfg=self.cfg.ema_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        super().load_managers()

        # Read the viewer flag here — USD stage is live at this point so
        # VisualizationMarkers can be safely constructed.
        # ManagerBasedRLEnv has no set_debug_vis hook, so this is the earliest
        # reliable place to create markers.

        # if self.cfg.debug_vis and self.sim.has_gui():
        #     self._map_vis_markers = mdp_markers.create_binary_map_markers()
        #     self._map_vis_last_map: torch.Tensor | None = None
        # else:
        #     self._map_vis_markers = None
        #     self._map_vis_last_map = None

    def step(self, action: torch.Tensor):
        obs, rew, terminated, truncated, info = super().step(action)

        # Call update_forbidden_markers for env 0 on every step, but only
        # re-upload geometry when the map actually changed (i.e. after a reset
        # of env 0). The hot path is a single torch.equal — negligible overhead.

        # if self._map_vis_markers is not None and hasattr(self, "plr_global_binary_map"):
        #     current_map = self.plr_global_binary_map[0]  # (H, W)
        #     if self._map_vis_last_map is None or not torch.equal(current_map, self._map_vis_last_map):
        #         mdp_markers.update_forbidden_markers(
        #             self._map_vis_markers,
        #             current_map,
        #             self.plr_map_origin_xy,
        #             float(self.plr_map_resolution),
        #         )
        #         self._map_vis_last_map = current_map.clone()

        return obs, rew, terminated, truncated, info
