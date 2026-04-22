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
parser.add_argument("--export_jit", action="store_true", default=False, help="Export policy as JIT module.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX model.")

# Binary-map playback
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

import plr_tasks.locomotion.velocity.mdp as mdp
import plr_tasks.locomotion.velocity.mdp.markers as mdp_markers


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


def load_checkpoint_with_fallback(
    runner: OnPolicyRunner,
    checkpoint_path: str,
    load_optimizer: bool = True,
) -> None:
    """Load checkpoint with compatibility handling."""
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    loaded_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    if load_optimizer:
        if runner.is_mdpo:
            runner.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
        else:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def export_policy_jit(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Export policy as JIT module to an 'export' folder next to the checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    actor_critic = runner.alg.actor_critic_1 if runner.is_mdpo else runner.alg.actor_critic
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    print(f"[INFO] Exporting JIT policy to: {export_dir}")
    actor_critic.export_jit(path=export_dir, filename="policy.pt", normalizer=normalizer)
    print("[INFO] JIT export complete!")


def export_policy_onnx(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Export policy as ONNX model to an 'export' folder next to the checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    actor_critic = runner.alg.actor_critic_1 if runner.is_mdpo else runner.alg.actor_critic
    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    if not hasattr(actor_critic, "export_onnx"):
        raise NotImplementedError(
            f"ONNX export not implemented for {type(actor_critic).__name__}. "
            "Please add an export_onnx method to this module."
        )

    print(f"[INFO] Exporting ONNX policy to: {export_dir}")
    actor_critic.export_onnx(path=export_dir, filename="policy.onnx", normalizer=normalizer)
    print("[INFO] ONNX export complete!")


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
    load_checkpoint_with_fallback(runner, resume_path)

    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)

    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # -------------------------------------------------------------------------
    # Binary-map visualization setup
    # -------------------------------------------------------------------------

    markers = mdp_markers.create_binary_map_markers()
    prev_global_map = None

    # Trace mode: restore one saved binary map snapshot.
    if args_cli.binary_map_trace_file is not None:
        trace = torch.load(args_cli.binary_map_trace_file)
        if len(trace) == 0:
            raise ValueError(f"Binary map trace file is empty: {args_cli.binary_map_trace_file}")

        entry = trace[args_cli.binary_map_trace_index]

        traced_map = entry["map"].to(base_env.device)
        traced_origin_xy = entry["map_origin_xy"].to(base_env.device)
        traced_res = float(entry["map_resolution"])

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

        mdp_markers.update_forbidden_markers(
            markers,
            base_env.plr_global_binary_map[0],
            base_env.plr_map_origin_xy,
            float(base_env.plr_map_resolution),
            z=0.10,
        )
        prev_global_map = base_env.plr_global_binary_map[0].detach().cpu().clone()

    # Live mode
    else:
        mdp_markers.update_forbidden_markers(
            markers,
            base_env.plr_global_binary_map[0],
            base_env.plr_map_origin_xy,
            float(base_env.plr_map_resolution),
            z=0.10,
        )
        prev_global_map = base_env.plr_global_binary_map[0].detach().cpu().clone()
        print("[play] live binary-map visualization enabled", flush=True)

    # First observations
    obs, _ = env.get_observations()

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        obs, _, _, _ = env.step(actions)

        # Update robot marker every frame
        root_pos = robot.data.root_pos_w[0:1, :3]
        root_quat = robot.data.root_quat_w[0:1, :]
        mdp_markers.update_robot_marker(markers, root_pos, root_quat)

        # Update forbidden markers only if the global map changed
        current_global_map = base_env.plr_global_binary_map[0].detach().cpu()
        if prev_global_map is None or not torch.equal(current_global_map, prev_global_map):
            mdp_markers.update_forbidden_markers(
                markers,
                base_env.plr_global_binary_map[0],
                base_env.plr_map_origin_xy,
                float(base_env.plr_map_resolution),
                z=0.10,
            )
            prev_global_map = current_global_map.clone()
            if args_cli.binary_map_trace_file is None:
                print("[play] global binary map changed", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
