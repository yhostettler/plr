# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from plr_tasks.locomotion.velocity.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class AnymalDFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
   
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    logger = "wandb"
    wandb_project = "plr_tasks_anymal_d"
    experiment_name = "anymal_navigation_ppo"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # rnn_hidden_size=256,
        # rnn_type="lstm_sru",
        # num_cameras=1,
        # image_input_dims=(64, 5, 8),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.1,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# @configclass
# class AnymalDFlatPPORunnerCfg(AnymalDRoughPPORunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 300
#         self.experiment_name = "anymal_d_flat"
#         self.logger = "tensorboard"



