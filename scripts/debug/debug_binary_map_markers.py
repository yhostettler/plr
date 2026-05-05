import argparse
import inspect

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Debug human-footstep binary-map updates and markers.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--steps", type=int, default=400, help="Interactive debug loop length.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that require SimulationApp
# -----------------------------------------------------------------------------

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import plr_tasks  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

import plr_tasks.locomotion.velocity.mdp as mdp
import plr_tasks.locomotion.velocity.mdp.markers as mdp_markers
from plr_tasks.locomotion.velocity.mdp.binary_map_cfg import BinaryMapLocalCfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def print_local_summary(local_map_flat: torch.Tensor, prefix: str = "") -> None:
    """Print shape, unique values, sum, and a center crop of the local map."""
    local_h = BinaryMapLocalCfg.LOCAL_H
    local_w = BinaryMapLocalCfg.LOCAL_W
    local_2d = local_map_flat.view(local_h, local_w)

    print(
        f"{prefix}local_shape={tuple(local_map_flat.shape)}, "
        f"local_unique={torch.unique(local_map_flat).cpu().numpy()}, "
        f"local_sum={local_map_flat.sum().item():.1f}",
        flush=True,
    )

    ch = local_h // 2
    cw = local_w // 2
    r0 = max(0, ch - 4)
    r1 = min(local_h, ch + 5)
    c0 = max(0, cw - 4)
    c1 = min(local_w, cw + 5)

    print("center 9x9:", flush=True)
    print(local_2d[r0:r1, c0:c1].cpu().numpy(), flush=True)


def print_global_summary(global_map_2d: torch.Tensor, prefix: str = "") -> None:
    """Print shape, unique values, and forbidden/allowed cell counts."""
    unique_vals = torch.unique(global_map_2d)
    num_forbidden = int((global_map_2d == 0).sum().item())
    num_allowed = int((global_map_2d == 1).sum().item())

    print(
        f"{prefix}global_shape={tuple(global_map_2d.shape)}, "
        f"global_unique={unique_vals.cpu().numpy()}, "
        f"forbidden_cells={num_forbidden}, "
        f"allowed_cells={num_allowed}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("creating env...", flush=True)

    print("mdp package file:", mdp.__file__, flush=True)
    print("binary_map_local module:", mdp.binary_map_local.__module__, flush=True)
    print("binary_map_local file:", inspect.getsourcefile(mdp.binary_map_local), flush=True)
    print("binary_map_local first line:", inspect.getsourcelines(mdp.binary_map_local)[1], flush=True)

    print("task in registry:", args_cli.task in gym.registry, flush=True)
    print([k for k in gym.registry.keys() if "PLR" in k or "B2W" in k or "Velocity" in k], flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()


    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    print("base_env.num_envs:", base_env.num_envs, flush=True)
    print("robot root_pos_w shape:", tuple(robot.data.root_pos_w.shape), flush=True)



    # Print map bounds in world coordinates
    origin = base_env.plr_map_origin_xy.detach().cpu()
    res = float(base_env.plr_map_resolution)
    h, w = base_env.plr_global_binary_map.shape[1:]
    x_min = origin[0].item()
    y_min = origin[1].item()
    x_max = x_min + w * res
    y_max = y_min + h * res

    print(f"map bounds world: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]", flush=True)



    

    print("obs space:", env.observation_space, flush=True)
    print("action space:", env.action_space, flush=True)

    # First snapshot
    local_map = mdp.binary_map_local(base_env)[0].detach().cpu()
    global_map = base_env.plr_global_binary_map[0].detach().cpu()

    print_global_summary(global_map, prefix="[init] ")
    print_local_summary(local_map, prefix="[init] ")

    # Optional: inspect human state if available
    if hasattr(base_env, "plr_human_pos_xy"):
        print("human_pos_xy shape:", tuple(base_env.plr_human_pos_xy.shape), flush=True)
    if hasattr(base_env, "plr_human_yaw"):
        print("human_yaw shape:", tuple(base_env.plr_human_yaw.shape), flush=True)
    if hasattr(base_env, "plr_human_phase"):
        print("human_phase shape:", tuple(base_env.plr_human_phase.shape), flush=True)

    # Markers
    markers = mdp_markers.create_binary_map_markers()
    mdp_markers.update_forbidden_markers(
        markers,
        base_env.plr_global_binary_map[0],
        base_env.plr_map_origin_xy,
        float(base_env.plr_map_resolution),
        z=0.10,
    )
    print("created markers", flush=True)
    print("entering interactive debug loop", flush=True)

    prev_local_map = local_map.clone()
    prev_global_map = global_map.clone()
    prev_global_map_vis = global_map.clone()

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.steps:
        actions = torch.zeros(env.action_space.shape, device=base_env.device)
        env.step(actions)

        local_map_k = mdp.binary_map_local(base_env)[0].detach().cpu()
        global_map_k = base_env.plr_global_binary_map[0].detach().cpu()

        root_pos = robot.data.root_pos_w[0:1, :3]
        root_quat = robot.data.root_quat_w[0:1, :]
        mdp_markers.update_robot_marker(markers, root_pos, root_quat)

        local_changed = not torch.equal(local_map_k, prev_local_map)
        num_local_changed = int((local_map_k != prev_local_map).sum().item())

        global_changed = not torch.equal(global_map_k, prev_global_map)
        num_global_changed = int((global_map_k != prev_global_map).sum().item())

        if global_changed:
            mdp_markers.update_forbidden_markers(
                markers,
                base_env.plr_global_binary_map[0],
                base_env.plr_map_origin_xy,
                float(base_env.plr_map_resolution),
                z=0.10,
            )
            prev_global_map_vis = global_map_k.clone()

        root_xy_k = robot.data.root_pos_w[0, :2].detach().cpu()

        if step_count < 10 or step_count % 25 == 0 or local_changed or global_changed:
            print(
                f"step {step_count:04d}: "
                f"root_xy={root_xy_k.numpy()}, "
                f"local_changed={local_changed}, "
                f"local_changed_cells={num_local_changed}, "
                f"global_changed={global_changed}, "
                f"global_changed_cells={num_global_changed}",
                flush=True,
            )

        if step_count < 5 or global_changed or step_count % 50 == 0:
            print_global_summary(global_map_k, prefix=f"[step {step_count:04d}] ")
            print_local_summary(local_map_k, prefix=f"[step {step_count:04d}] ")

            if hasattr(base_env, "plr_human_pos_xy"):
                print(
                    f"[step {step_count:04d}] first human positions env0:\n"
                    f"{base_env.plr_human_pos_xy[0].detach().cpu().numpy()}",
                    flush=True,
                )
            if hasattr(base_env, "plr_human_phase"):
                print(
                    f"[step {step_count:04d}] first human phases env0:\n"
                    f"{base_env.plr_human_phase[0].detach().cpu().numpy()}",
                    flush=True,
                )
            if hasattr(base_env, "plr_human_step_timer"):
                print(
                    f"[step {step_count:04d}] first human step timers env0:\n"
                    f"{base_env.plr_human_step_timer[0].detach().cpu().numpy()}",
                    flush=True,
                )

        prev_local_map = local_map_k.clone()
        prev_global_map = global_map_k.clone()

        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
