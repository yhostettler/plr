import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug binary map markers in Isaac Lab.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    import inspect
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import plr_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import plr_tasks.locomotion.velocity.mdp as mdp
    import plr_tasks.locomotion.velocity.mdp.markers as mdp_markers

    print("mdp package file:", mdp.__file__, flush=True)
    print("binary_map_2x2 module:", mdp.binary_map_2x2.__module__, flush=True)
    print("binary_map_2x2 file:", inspect.getsourcefile(mdp.binary_map_2x2), flush=True)
    print("binary_map_2x2 first line:", inspect.getsourcelines(mdp.binary_map_2x2)[1], flush=True)

    print("task in registry:", args_cli.task in gym.registry, flush=True)
    print([k for k in gym.registry.keys() if "PLR" in k or "B2W" in k or "Velocity" in k], flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    print("created env", flush=True)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]


    # Trigger lazy init of the global map and sampled indices.
    ego_map = mdp.binary_map_2x2(base_env)[0].detach().cpu()
    global_map = base_env.plr_global_binary_map[0].detach().cpu()

    print("global map shape:", tuple(global_map.shape), flush=True)
    print("global map unique values:", torch.unique(global_map), flush=True)
    print("ego map at reset [FL, FR, RL, RR]:", ego_map, flush=True)

    markers = mdp_markers.create_binary_map_markers()
    print("created markers", flush=True)

    # Draw forbidden cells once because the map is static for now.
    mdp_markers.update_forbidden_markers(
        markers,
        base_env.plr_global_binary_map[0],
        base_env.plr_map_origin_xy,
        float(base_env.plr_map_resolution),
        z=0.10,
    )

    print("entering interactive debug loop", flush=True)

    k = 0
    while simulation_app.is_running():
        ego_map_k = mdp.binary_map_2x2(base_env)[0].detach().cpu()

        # Robot marker
        root_pos = robot.data.root_pos_w[0:1, :3]
        root_quat = robot.data.root_quat_w[0:1, :]
        mdp_markers.update_robot_marker(markers, root_pos, root_quat)

        # Sampled egocentric cells
        rows = base_env.plr_last_rows[0]
        cols = base_env.plr_last_cols[0]
        if k % 20 == 0:
            mdp_markers.update_forbidden_markers(
                markers,
                base_env.plr_global_binary_map[0],
                base_env.plr_map_origin_xy,
                float(base_env.plr_map_resolution),
                z=0.10,
        )



        if k < 5 or k % 100 == 0:
            root_xy_k = robot.data.root_pos_w[0, :2].detach().cpu()
            print(
                f"step {k}: "
                f"root_xy={root_xy_k.numpy()}, "
                f"rows={rows.detach().cpu().tolist()}, "
                f"cols={cols.detach().cpu().tolist()}, "
                f"ego={ego_map_k.numpy()}",
                flush=True,
            )

        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=base_env.device)
            env.step(actions)

        simulation_app.update()
        k += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
