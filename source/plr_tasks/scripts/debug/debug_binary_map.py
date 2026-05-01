import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug global and egocentric binary map.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
import plr_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


import plr_tasks.terrain_locomotion.mdp as mdp


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()



    base_env = env.unwrapped



    # 1) global map
    ego_map = mdp.binary_map_2x2(base_env)[0].detach().cpu()

    global_map = base_env.plr_global_binary_map[0].detach().cpu()
    print("global map shape:", tuple(global_map.shape))
    print("global map unique values:", torch.unique(global_map))

    plt.figure()
    plt.imshow(global_map.numpy(), origin="lower")
    plt.title("Global binary map, env 0")
    plt.colorbar()
    plt.show(block=False)

    # 2) current robot pose
    robot = base_env.scene["robot"]
    root_xy = robot.data.root_pos_w[0, :2].detach().cpu()
    root_quat = robot.data.root_quat_w[0].detach().cpu()
    print("root_xy:", root_xy)
    print("root_quat_wxyz:", root_quat)
    print("obs space:", env.observation_space)
    print("action space:", env.action_space)

    # 3) egocentric map from your function directly
    print("ego map at reset [FL, FR, RL, RR]:", ego_map)

    # 4) step a few times with zero action
    for k in range(1000):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=base_env.device)
            env.step(actions)

        ego_map_k = mdp.binary_map_2x2(base_env)[0].detach().cpu()
        root_xy_k = robot.data.root_pos_w[0, :2].detach().cpu()
        print(f"step {k:02d}: root_xy={root_xy_k.numpy()}, ego={ego_map_k.numpy()}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
