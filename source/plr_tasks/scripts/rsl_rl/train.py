"""Train a PLR locomotion / navigation policy with RSL-RL.

This version:
- uses the task registry to construct env and agent configs
- supports optional video recording
- writes logs to logs/rsl_rl/<experiment_name>/<timestamp>
- saves binary_map_trace.pt at the end if the env collected one
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train a policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of each recorded video in steps.")
parser.add_argument(
    "--video_interval",
    type=int,
    default=20000,
    help="Record a video every N environment steps.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of learning iterations.")
parser.add_argument("--run_name", type=str, default=None, help="Name of the wandb run (appended to log directory).")
parser.add_argument("--debug_vis", action="store_true", default=False, help="Draw binary map forbidden cells in the viewport during training.")

# Isaac Lab app args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras when video is requested
if args_cli.video:
    args_cli.enable_cameras = True


# -----------------------------------------------------------------------------
# Launch app BEFORE importing Isaac Sim dependent modules
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that require the app
# -----------------------------------------------------------------------------

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import plr_tasks  # noqa: F401

# Register PLR-specific network architectures so that OnPolicyRunner's eval()
# can resolve them by name. eval() runs in on_policy_runner's global namespace,
# so we inject directly into that module's __dict__.
import rsl_rl.runners.on_policy_runner as _runner_module
from plr_tasks.modules import ActorCriticRecurrentWithMapEncoder  # noqa: F401

_runner_module.ActorCriticRecurrentWithMapEncoder = ActorCriticRecurrentWithMapEncoder

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    """Train with RSL-RL."""
    if args_cli.task is None:
        raise ValueError("Please provide --task")

    # Get config entry points from the Gym registry
    spec = gym.spec(args_cli.task)
    env_cfg_class = spec.kwargs.get("env_cfg_entry_point")
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    if env_cfg_class is None:
        raise ValueError(f"Task '{args_cli.task}' does not provide env_cfg_entry_point")
    if agent_cfg_class is None:
        raise ValueError(f"Task '{args_cli.task}' does not provide rsl_rl_cfg_entry_point")

    env_cfg: ManagerBasedRLEnvCfg = env_cfg_class()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()

    # -------------------------------------------------------------------------
    # Override configs from CLI
    # -------------------------------------------------------------------------

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.debug_vis:
        env_cfg.debug_vis = True

    if args_cli.seed is not None:
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args_cli.seed
        if hasattr(agent_cfg, "seed"):
            agent_cfg.seed = args_cli.seed

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
        
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Task: {args_cli.task}", flush=True)
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}", flush=True)
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}", flush=True)
    print(f"[INFO] Log dir: {log_dir}", flush=True)

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)

        env = RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording videos to: {video_folder}", flush=True)

    env = RslRlVecEnvWrapper(env)

    # -------------------------------------------------------------------------
    # Runner
    # -------------------------------------------------------------------------

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    # -------------------------------------------------------------------------
    # Save binary-map trace if the environment collected one
    # -------------------------------------------------------------------------

    base_env = env.unwrapped
    if hasattr(base_env, "plr_binary_map_trace"):
        trace_path = os.path.join(log_dir, "binary_map_trace.pt")
        torch.save(base_env.plr_binary_map_trace, trace_path)
        print(f"[binary map trace] saved to {trace_path}", flush=True)



    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

