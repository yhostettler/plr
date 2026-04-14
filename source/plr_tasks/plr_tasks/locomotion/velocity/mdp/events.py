import torch
from isaaclab.envs import ManagerBasedRLEnv


def reset_binary_map_2x2(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Sample a fresh 2x2 binary map for the resetting environments."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "plr_binary_map_2x2"):
        env.plr_binary_map_2x2 = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)

    env.plr_binary_map_2x2[env_ids] = torch.randint(
        low=0,
        high=2,
        size=(len(env_ids), 4),
        device=env.device,
        dtype=torch.int64,
    ).to(torch.float32)