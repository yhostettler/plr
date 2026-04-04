# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Isaac Lab Navigation Tasks extension.

This extension provides navigation tasks for robot learning with visual perception.

On import, this module:
1. Applies monkey-patches to Isaac Lab terrain system for height field storage
2. Registers maze terrain types
3. Registers navigation task environments
"""

import os
import toml

# Conveniences to other module directories via relative paths
PLR_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

PLR_TASKS_METADATA = toml.load(os.path.join(PLR_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = PLR_TASKS_METADATA["package"]["version"]


##
# Register Gym environments.
##

import plr_tasks.locomotion.velocity.config.anymal_d


from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", "terrains"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
