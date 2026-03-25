import torch
from isaaclab.envs import ManagerBasedRLEnv


def fixed_binary_2x2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a fixed 2x2 binary map for every environment.

    Convention:
        0 = forbidden
        1 = allowed

    Cell order:
        [front_left, front_right, rear_left, rear_right]
    """
    pattern = torch.tensor([0.0, 1.0, 1.0, 0.0], device=env.device, dtype=torch.float32)
    return pattern.unsqueeze(0).repeat(env.num_envs, 1)
