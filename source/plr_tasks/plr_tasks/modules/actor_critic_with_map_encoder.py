# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab).
# All rights reserved.
# SPDX-License-Identifier: MIT

"""CNN-based binary map encoder plugged into ActorCriticRecurrent."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent


class BinaryMapCNN(nn.Module):
    """Encodes a binary occupancy map (32×32) into a compact feature vector.

    Expects input shape ``(B, 1, 32, 32)``. One stride-1 entry conv followed by
    three stride-2 convs reduce spatial dims to 4×4 (1024 flat), then a linear
    head projects to ``enc_dim``:

    1→16 (s1) → 16→32 (s2) → 32→64 (s2) → 64→64 (s2) → flat 1024 → enc_dim

    Args:
        enc_dim: Dimension of the output feature vector.
        activation: Activation function name (forwarded to ``get_activation``).
    """

    def __init__(self, enc_dim: int, activation: str = "elu"):
        super().__init__()
        act = get_activation(activation)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), act,   # 32→32  entry conv
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), act,  # 32→16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), act,  # 16→8
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), act,  #  8→4
        )
        # 64 channels * 4x4 spatial = 1024 flat
        self.fc = nn.Linear(64 * 4 * 4, enc_dim)
        self.fc_act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map tensor ``(B, 1, 32, 32)`` → feature vector ``(B, enc_dim)``."""
        # conv → flatten → project to enc_dim
        return self.fc_act(self.fc(self.conv(x).flatten(1)))


class ActorCriticRecurrentWithMapEncoder(ActorCriticRecurrent):
    """Recurrent Actor-Critic with a per-step CNN encoder for binary map observations.

    The observation vector must have the flattened binary map as its **last**
    ``map_obs_dim`` elements.  At every forward pass those elements are
    reshaped to ``(B, 1, map_height, map_width)``, encoded by a
    :class:`BinaryMapCNN`, and the resulting ``map_enc_dim`` features are
    concatenated with the remaining proprioceptive features before being fed
    into the LSTM.

    This reduces the LSTM input from
    ``num_proprio + map_obs_dim`` to ``num_proprio + map_enc_dim``
    (e.g. 304 → 80 for the default AnymalD setup) while the CNN preserves
    spatial structure that a flat MLP cannot exploit.

    Actor and critic each have their own CNN so they can learn different map
    representations without interference.

    Args:
        num_actor_obs: Total actor observation dimension (proprio + flat map).
        num_critic_obs: Total critic observation dimension (proprio + flat map).
        num_actions: Number of action dimensions.
        map_obs_dim: Number of elements occupied by the flat map in the obs vector.
        map_height: Height of the 2-D binary map grid.
        map_width: Width of the 2-D binary map grid.
        map_enc_dim: Output dimension of the CNN encoder.
        All remaining keyword arguments are forwarded to
        :class:`~rsl_rl.modules.ActorCriticRecurrent`.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        map_obs_dim: int = 1024,
        map_height: int = 32,
        map_width: int = 32,
        map_enc_dim: int = 64,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        rnn_type: str = "lstm_sru",
        dropout: float = 0.0,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        num_proprio_actor = num_actor_obs - map_obs_dim
        num_proprio_critic = num_critic_obs - map_obs_dim

        if num_proprio_actor <= 0 or num_proprio_critic <= 0:
            raise ValueError(
                f"map_obs_dim ({map_obs_dim}) must be smaller than "
                f"num_actor_obs ({num_actor_obs}) / num_critic_obs ({num_critic_obs})."
            )
        if map_height * map_width != map_obs_dim:
            raise ValueError(
                f"map_height ({map_height}) * map_width ({map_width}) = "
                f"{map_height * map_width} != map_obs_dim ({map_obs_dim})."
            )

        super().__init__(
            num_actor_obs=num_proprio_actor + map_enc_dim,
            num_critic_obs=num_proprio_critic + map_enc_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            dropout=dropout,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        self.map_obs_dim = map_obs_dim
        self.map_enc_dim = map_enc_dim
        self.map_height = map_height
        self.map_width = map_width

        self.map_encoder_actor = BinaryMapCNN(map_enc_dim, activation)
        self.map_encoder_critic = BinaryMapCNN(map_enc_dim, activation)

        print(
            f"[ActorCriticRecurrentWithMapEncoder] "
            f"CNN: {map_height}×{map_width} ({map_obs_dim} flat) → {map_enc_dim} features"
        )
        print(
            f"  Actor  LSTM input: {num_proprio_actor} proprio + {map_enc_dim} map "
            f"= {num_proprio_actor + map_enc_dim}"
        )
        print(
            f"  Critic LSTM input: {num_proprio_critic} proprio + {map_enc_dim} map "
            f"= {num_proprio_critic + map_enc_dim}"
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _encode_obs(self, observations: torch.Tensor, map_encoder: BinaryMapCNN) -> torch.Tensor:
        """Replace the flat map slice with CNN features.

        Works for both inference shape ``(B, D)`` and batch-training shape
        ``(L, B, D)`` (the latter comes from the LSTM rollout storage).
        """
        is_seq = observations.dim() == 3  # (L, B, D) during PPO update
        if is_seq:
            L, B, _ = observations.shape
            obs_2d = observations.reshape(L * B, -1)
        else:
            obs_2d = observations

        proprio = obs_2d[..., : -self.map_obs_dim]
        map_2d = obs_2d[..., -self.map_obs_dim :].view(-1, 1, self.map_height, self.map_width)
        encoded = torch.cat([proprio, map_encoder(map_2d)], dim=-1)

        if is_seq:
            encoded = encoded.reshape(L, B, -1)
        return encoded

    # ------------------------------------------------------------------
    # Forward overrides
    # ------------------------------------------------------------------

    def act(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        return super().act(self._encode_obs(observations, self.map_encoder_actor), masks, hidden_states, dropout_masks)

    def act_inference(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        return super().act_inference(
            self._encode_obs(observations, self.map_encoder_actor), masks, hidden_states, dropout_masks
        )

    def evaluate(self, critic_observations, masks=None, hidden_states=None, dropout_masks=None):
        return super().evaluate(
            self._encode_obs(critic_observations, self.map_encoder_critic), masks, hidden_states, dropout_masks
        )

    # ------------------------------------------------------------------
    # Parameter groups — include CNN weights in actor / critic optimiser groups
    # ------------------------------------------------------------------

    def get_actor_parameters(self):
        return super().get_actor_parameters() + list(self.map_encoder_actor.parameters())

    def get_critic_parameters(self):
        return super().get_critic_parameters() + list(self.map_encoder_critic.parameters())

    # ------------------------------------------------------------------
    # Export stubs
    # ------------------------------------------------------------------

    def export_jit(self, path: str, filename: str = "policy.pt", normalizer=None):
        raise NotImplementedError(
            "JIT export is not yet implemented for ActorCriticRecurrentWithMapEncoder."
        )

    def export_onnx(self, path: str, filename: str = "policy.onnx", normalizer=None, num_obs: int = None):
        raise NotImplementedError(
            "ONNX export is not yet implemented for ActorCriticRecurrentWithMapEncoder."
        )
