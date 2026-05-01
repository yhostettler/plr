import argparse
import inspect
import time

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Debug binary-map sampling, updates, markers, and performance.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--steps", type=int, default=400, help="Interactive debug loop length.")
parser.add_argument("--bench_calls", type=int, default=200, help="Number of calls for binary_map_local benchmark.")
parser.add_argument("--bench_steps", type=int, default=200, help="Number of steps for env.step benchmark.")
parser.add_argument("--skip_bench", action="store_true", default=False, help="Skip timing benchmarks.")
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

import plr_tasks.terrain_locomotion.mdp as mdp
import plr_tasks.terrain_locomotion.mdp.markers as mdp_markers
from plr_tasks.terrain_locomotion.mdp.binary_map_cfg import BinaryMapLocalCfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def benchmark_binary_map_local(base_env, num_calls: int) -> None:
    """Benchmark the local binary-map sampler only."""
    print("\n[bench] binary_map_local()", flush=True)

    # warmup
    for _ in range(50):
        _ = mdp.binary_map_local(base_env)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_calls):
        _ = mdp.binary_map_local(base_env)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    ms_per_call = (t1 - t0) / num_calls * 1000.0
    us_per_env = ms_per_call / base_env.num_envs * 1000.0

    print(f"[bench] device           : {base_env.device}", flush=True)
    print(f"[bench] num_envs         : {base_env.num_envs}", flush=True)
    print(f"[bench] local map size   : {BinaryMapLocalCfg.LOCAL_H}x{BinaryMapLocalCfg.LOCAL_W}", flush=True)
    print(f"[bench] ms / call        : {ms_per_call:.3f}", flush=True)
    print(f"[bench] us / env / call  : {us_per_env:.3f}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = mdp.binary_map_local(base_env)
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[bench] cuda allocatedMB : {allocated_mb:.2f}", flush=True)
        print(f"[bench] cuda peakMB      : {peak_mb:.2f}", flush=True)


def benchmark_env_step(env, base_env, num_steps: int) -> None:
    """Benchmark full env.step() cost with zero actions."""
    print("\n[bench] env.step()", flush=True)

    # warmup
    for _ in range(50):
        actions = torch.zeros(env.action_space.shape, device=base_env.device)
        env.step(actions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_steps):
        actions = torch.zeros(env.action_space.shape, device=base_env.device)
        env.step(actions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    ms_per_step = (t1 - t0) / num_steps * 1000.0
    us_per_env = ms_per_step / base_env.num_envs * 1000.0

    print(f"[bench] ms / env.step     : {ms_per_step:.3f}", flush=True)
    print(f"[bench] us / env / step  : {us_per_env:.3f}", flush=True)


def print_local_summary(local_map_flat: torch.Tensor, prefix: str = "") -> None:
    """Print shape, unique values, local sum, and center crop."""
    local_h = BinaryMapLocalCfg.LOCAL_H
    local_w = BinaryMapLocalCfg.LOCAL_W
    local_2d = local_map_flat.view(local_h, local_w)

    print(
        f"{prefix}local_shape={tuple(local_map_flat.shape)}, "
        f"local_unique={torch.unique(local_map_flat).numpy()}, "
        f"local_sum={local_map_flat.sum().item():.1f}",
        flush=True,
    )

    # center 9x9 crop
    ch = local_h // 2
    cw = local_w // 2
    r0 = ch - 4
    r1 = ch + 5
    c0 = cw - 4
    c1 = cw + 5
    print("center 9x9:", flush=True)
    print(local_2d[r0:r1, c0:c1].numpy(), flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("created env", flush=True)
    print("created markers", flush=True)
    print("entering loop", flush=True)

    print("mdp package file:", mdp.__file__, flush=True)
    print("binary_map_local module:", mdp.binary_map_local.__module__, flush=True)
    print("binary_map_local file:", inspect.getsourcefile(mdp.binary_map_local), flush=True)
    print("binary_map_local first line:", inspect.getsourcelines(mdp.binary_map_local)[1], flush=True)

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

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    print("obs space:", env.observation_space, flush=True)
    print("action space:", env.action_space, flush=True)

    # First snapshot
    local_map = mdp.binary_map_local(base_env)[0].detach().cpu()
    global_map = base_env.plr_global_binary_map[0].detach().cpu()

    print("global map shape:", tuple(base_env.plr_global_binary_map.shape), flush=True)
    print("global map unique values:", torch.unique(base_env.plr_global_binary_map[0]), flush=True)
    print("local map shape:", tuple(local_map.shape), flush=True)
    print("local map unique values:", torch.unique(local_map), flush=True)
    print_local_summary(local_map)

    # Benchmarks
    if not args_cli.skip_bench:
        benchmark_binary_map_local(base_env, args_cli.bench_calls)
        benchmark_env_step(env, base_env, args_cli.bench_steps)

        # Reset after stepping benchmark so the interactive loop starts fresh
        env.reset()
        local_map = mdp.binary_map_local(base_env)[0].detach().cpu()
        global_map = base_env.plr_global_binary_map[0].detach().cpu()

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

    prev_local_map = None
    prev_global_map = None
    prev_global_map_vis = None

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.steps:
        actions = torch.zeros(env.action_space.shape, device=base_env.device)
        env.step(actions)

        # Current snapshots
        local_map_k = mdp.binary_map_local(base_env)[0].detach().cpu()
        global_map_k = base_env.plr_global_binary_map[0].detach().cpu()

        # Update robot marker every step
        root_pos = robot.data.root_pos_w[0:1, :3]
        root_quat = robot.data.root_quat_w[0:1, :]
        mdp_markers.update_robot_marker(markers, root_pos, root_quat)

        # Redraw forbidden markers if the global map changed
        if prev_global_map_vis is None or not torch.equal(global_map_k, prev_global_map_vis):
            t0 = time.time()
            mdp_markers.update_forbidden_markers(
                markers,
                base_env.plr_global_binary_map[0],
                base_env.plr_map_origin_xy,
                float(base_env.plr_map_resolution),
                z=0.10,
            )
            redraw_ms = (time.time() - t0) * 1000.0
            print(f"step {step_count:04d}: redrew forbidden markers ({redraw_ms:.3f} ms)", flush=True)
            prev_global_map_vis = global_map_k.clone()

        # Periodic diagnostics
        if step_count == 0:
            prev_local_map = local_map_k.clone()
            prev_global_map = global_map_k.clone()

        if step_count < 5 or step_count % 100 == 0:
            root_xy_k = robot.data.root_pos_w[0, :2].detach().cpu()
            print(
                f"step {step_count:04d}: root_xy={root_xy_k.numpy()}, ",
                end="",
                flush=True,
            )
            print_local_summary(local_map_k)

        if step_count % 50 == 0 and step_count > 0:
            local_changed = not torch.equal(local_map_k, prev_local_map)
            num_local_changed = int((local_map_k != prev_local_map).sum().item())

            global_changed = not torch.equal(global_map_k, prev_global_map)
            num_global_changed = int((global_map_k != prev_global_map).sum().item())

            print(
                f"step {step_count:04d}: local map changed = {local_changed}, changed cells = {num_local_changed}",
                flush=True,
            )
            print(
                f"step {step_count:04d}: global map changed = {global_changed}, changed cells = {num_global_changed}",
                flush=True,
            )

            prev_local_map = local_map_k.clone()
            prev_global_map = global_map_k.clone()

        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

