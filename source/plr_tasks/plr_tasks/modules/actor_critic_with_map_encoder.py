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
    """Encodes a flat binary occupancy map into a compact feature vector.

    The flat map of ``map_h * map_w`` values is reshaped to ``(B, 1, map_h, map_w)``
    and passed through an adaptive convolutional stack: one stride-1 entry conv
    followed by stride-2 convs that halve spatial dims until reaching ≤ 2×2.
    The result is then projected to ``enc_dim`` by a linear layer.

    Channels are doubled at each stride-2 step and capped at 64:

    - 16×16:  1→8 (s1) → 8→16 (s2) → 16→32 (s2) → 32→64 (s2) → 2×2 → flat 256 → enc_dim
    - 32×32:  1→8 (s1) → 8→16 (s2) → 16→32 (s2) → 32→64 (s2) → 64→64 (s2) → 2×2 → flat 256 → enc_dim
    - 64×64:  1→8 (s1) → 8→16 (s2) → 16→32 (s2) → 32→64 (s2) → 64→64 (s2) → 64→64 (s2) → 64→64 (s2) → 2×2 → flat 256 → enc_dim

    The linear head is always 256 → ``enc_dim`` regardless of map size.

    Args:
        map_h: Height of the 2-D binary map.
        map_w: Width of the 2-D binary map.
        enc_dim: Dimension of the output feature vector.
        activation: Activation function name (forwarded to ``get_activation``).
    """

    def __init__(self, map_h: int, map_w: int, enc_dim: int, activation: str = "elu"):
        super().__init__()
        self.map_h = map_h
        self.map_w = map_w

        layers: list[nn.Module] = []
        in_ch, out_ch = 1, 8
        h, w = map_h, map_w

        # Stride-1 entry conv
        layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), get_activation(activation)]
        in_ch = out_ch

        # Stride-2 convs until spatial size reaches ≤ 2×2
        while h > 2 or w > 2:
            out_ch = min(in_ch * 2, 64)
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1), get_activation(activation)]
            h = (h + 1) // 2
            w = (w + 1) // 2
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch * h * w, enc_dim)
        self.fc_act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map flat tensor ``(B, map_h * map_w)`` → feature vector ``(B, enc_dim)``."""
        B = x.shape[0]
        x = x.view(B, 1, self.map_h, self.map_w)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc_act(self.fc(x))


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
        map_obs_dim: int = 4096,
        map_height: int = 64,
        map_width: int = 64,
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

        self.map_encoder_actor = BinaryMapCNN(map_height, map_width, map_enc_dim, activation)
        self.map_encoder_critic = BinaryMapCNN(map_height, map_width, map_enc_dim, activation)

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
        map_flat = obs_2d[..., -self.map_obs_dim :]
        encoded = torch.cat([proprio, map_encoder(map_flat)], dim=-1)

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
