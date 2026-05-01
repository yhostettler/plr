"""Play a trained navigation policy (PPO/MDPO) with automatic checkpoint loading.

Usage:
   python scripts/play.py --task <task_name> [options]

Arguments:
   --task                   Task name
   --checkpoint             Path to model checkpoint (.pt file)
   --use_last_checkpoint    Use latest checkpoint from logs (default behavior)
   --num_envs               Number of parallel environments
   --video                  Enable video recording
   --video_length           Video length in steps (default: 200)
   --binary_map_trace_file  Path to binary_map_trace.pt
   --binary_map_trace_index Index into trace list (default: -1 = last)

Modes:
   - Live mode  : no --binary_map_trace_file
                  -> binary map comes from the live environment and updates during play
   - Trace mode : with --binary_map_trace_file
                  -> restore one logged binary-map snapshot and visualize it
"""

from __future__ import annotations

import argparse
import os
import re
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Play a trained navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint from logs.")

# Binary-map playback
parser.add_argument("--debug_vis", action="store_true", default=False, help="Draw binary map forbidden cells in the viewport (live map, no trace needed).")
parser.add_argument("--binary_map_trace_file", type=str, default=None, help="Path to binary_map_trace.pt")
parser.add_argument("--binary_map_trace_index", type=int, default=-1, help="Trace entry index (-1 = last)")

# Isaac Lab app args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras for play/visualization
args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch app BEFORE importing Isaac Sim dependent modules
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that require the app
# -----------------------------------------------------------------------------

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import plr_tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import plr_tasks.terrain_locomotion.mdp as mdp
import plr_tasks.terrain_locomotion.mdp.markers as mdp_markers


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = r"model_.*\.pt") -> str:
    """Find the latest checkpoint file in the latest run directory."""
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs: list[str] = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    checkpoint_files: list[str] = []
    for filename in os.listdir(run_path):
        if re.match(checkpoint_pattern, filename):
            checkpoint_files.append(filename)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    checkpoint_files.sort(key=lambda name: f"{name:0>15}")
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(run_path, latest_checkpoint)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    """Play navigation policy with RSL-RL and binary-map visualization."""
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.debug_vis:
        env_cfg.debug_vis = True

    # In trace mode, disable binary-map events so the restored map is not overwritten.
    if args_cli.binary_map_trace_file is not None and hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "binary_map_reset"):
            env_cfg.events.binary_map_reset = None
        if hasattr(env_cfg.events, "binary_map_interval_update"):
            env_cfg.events.binary_map_interval_update = None

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    env = RslRlVecEnvWrapper(env)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern=r"model_.*\.pt")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # -------------------------------------------------------------------------
    # Binary-map visualization setup
    # -------------------------------------------------------------------------
    # Two mutually exclusive modes:
    #
    #   --debug_vis              Live mode: env.step() owns the markers and redraws
    #                            them automatically on each map reset. Shows the real
    #                            random patches the policy faces during play.
    #
    #   --binary_map_trace_file  Trace mode: inject one logged map snapshot, freeze
    #                            the map events, and manage markers here in the script.
    #                            Useful for reproducing a specific training situation.

    trace_markers = None
    trace_prev_map = None

    if args_cli.binary_map_trace_file is not None:
        trace = torch.load(args_cli.binary_map_trace_file)
        if len(trace) == 0:
            raise ValueError(f"Binary map trace file is empty: {args_cli.binary_map_trace_file}")

        entry = trace[args_cli.binary_map_trace_index]

        traced_map = entry["map"].to(base_env.device)
        traced_origin_xy = entry["map_origin_xy"].to(base_env.device)
        traced_res = float(entry["map_resolution"])

        # Inject the logged map into every env so observations stay consistent.
        if (
            not hasattr(base_env, "plr_global_binary_map")
            or base_env.plr_global_binary_map.shape[1:] != traced_map.shape
        ):
            base_env.plr_global_binary_map = traced_map.unsqueeze(0).repeat(base_env.num_envs, 1, 1)
        else:
            base_env.plr_global_binary_map[:] = traced_map.unsqueeze(0)

        base_env.plr_map_origin_xy = traced_origin_xy
        base_env.plr_map_resolution = traced_res

        print(
            f"[play] loaded binary map trace entry {args_cli.binary_map_trace_index} "
            f"from {args_cli.binary_map_trace_file}",
            flush=True,
        )
        print(
            f"[play] event={entry['event']} step={entry['step']} env_id={entry['env_id']}",
            flush=True,
        )

        # Create and draw markers for the frozen trace map.
        trace_markers = mdp_markers.create_binary_map_markers()
        mdp_markers.update_forbidden_markers(
            trace_markers,
            base_env.plr_global_binary_map[0],
            base_env.plr_map_origin_xy,
            float(base_env.plr_map_resolution),
            z=0.10,
        )
        trace_prev_map = base_env.plr_global_binary_map[0].detach().cpu().clone()

    # First observations
    obs, _ = env.get_observations()

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        obs, _, _, _ = env.step(actions)

        # Trace mode: update robot marker and redraw forbidden cells if the map
        # somehow changed (should stay frozen, but defensive check is cheap).
        if trace_markers is not None:
            root_pos = robot.data.root_pos_w[0:1, :3]
            root_quat = robot.data.root_quat_w[0:1, :]
            mdp_markers.update_robot_marker(trace_markers, root_pos, root_quat)

            current_global_map = base_env.plr_global_binary_map[0].detach().cpu()
            if not torch.equal(current_global_map, trace_prev_map):
                mdp_markers.update_forbidden_markers(
                    trace_markers,
                    base_env.plr_global_binary_map[0],
                    base_env.plr_map_origin_xy,
                    float(base_env.plr_map_resolution),
                    z=0.10,
                )
                trace_prev_map = current_global_map.clone()

        # Live mode: env.step() handles forbidden-cell markers automatically
        # when cfg.debug_vis=True (set via --debug_vis).

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
