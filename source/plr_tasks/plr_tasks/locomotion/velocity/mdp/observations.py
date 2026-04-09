import csv
from pathlib import Path

import torch
from isaaclab.envs import ManagerBasedRLEnv


def binary_map_2x2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the stored 2x2 egocentric binary map for every environment.

    Order:
        [front_left, front_right, rear_left, rear_right]

    Convention:
        0 = forbidden
        1 = allowed
    """
    if not hasattr(env, "plr_binary_map_2x2"):
        pattern = torch.tensor([0.0, 1.0, 1.0, 0.0], device=env.device, dtype=torch.float32)
        env.plr_binary_map_2x2 = pattern.unsqueeze(0).repeat(env.num_envs, 1)

    # lazy init for step logging
    if not hasattr(env, "plr_log_step_counter"):
        env.plr_log_step_counter = 0
        env.plr_log_path = Path("plr_binary_map_log.csv")
        with env.plr_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "front_left", "front_right", "rear_left", "rear_right"])

    env.plr_log_step_counter += 1

    # log only env 0 to keep file size manageable
    vals = env.plr_binary_map_2x2[0].detach().cpu().tolist()
    with env.plr_log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([env.plr_log_step_counter, *vals])

    return env.plr_binary_map_2x2