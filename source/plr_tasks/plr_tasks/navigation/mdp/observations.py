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

    return env.plr_binary_map_2x2
