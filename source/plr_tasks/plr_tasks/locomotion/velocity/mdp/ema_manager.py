import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .ema_manager_cfg import EMAManagerCfg

class EMAManager:
    """Manager for maintaining an Exponential Moving Average (EMA) of a tensor.

    This manager tracks a moving average of a given quantity (e.g., linear velocity)
    across simulation steps, providing a smoothed version for observations or rewards.
    """

    def __init__(self, cfg: "EMAManagerCfg", num_envs: int, device: str):
        """Initialize the EMA manager.

        Args:
            cfg: Configuration for the EMA manager.
            # num_envs: Number of environments.
            device: Device to perform operations on.
        """
        self.cfg = cfg
        # self.num_envs = num_envs
        self.device = device
        self._enabled = cfg.enabled
        self._alpha = cfg.alpha
        self._beta = cfg.beta
        self._ema_signal_state: torch.Tensor | None = None
        self._ema_error_state_lin_vel_xy: torch.Tensor | None = None
        self._ema_error_state_ang_vel_z: torch.Tensor | None = None

    def _init_ema_signal_state(self, shape: torch.Size):
        """Initialize the EMA signal state tensor."""
        self._ema_signal_state = torch.zeros(shape, device=self.device)

    def _init_ema_error_state_lin_vel_xy(self, shape: torch.Size):
        """Initialize the EMA error state tensor."""
        self._ema_error_state_lin_vel_xy = torch.zeros(shape, device=self.device)

    def _init_ema_error_state_ang_vel_z(self, shape: torch.Size):
        """Initialize the EMA error state tensor."""
        self._ema_error_state_ang_vel_z = torch.zeros(shape, device=self.device)

    def compute_ema_signal(self, data: torch.Tensor) -> torch.Tensor:
        """Update the EMA signal state with the new data and return the updated EMA.

        Args:
            data: The current data signal (actual) tensor of shape (num_envs, ...).

        Returns:
            The updated EMA tensor. If EMA is disabled, returns the original data.
        """
        if not self._enabled:
            return data
        

        if self._ema_signal_state is None or self._ema_signal_state.shape != data.shape:
            self._init_ema_signal_state(data.shape)
            self._ema_signal_state.copy_(data)
            return self._ema_signal_state

        if data.shape[-1] == 3:
            # Apply alpha to x, y and beta to z
            # data shape is (num_envs, ..., 3)
            # We create a coefficient tensor of shape (3,) and it will broadcast
            coeffs = torch.tensor([self._alpha, self._alpha, self._beta], device=self.device, dtype=data.dtype)
            self._ema_signal_state = coeffs * data + (1.0 - coeffs) * self._ema_signal_state
        else:
            # Fallback to single alpha for other shapes
            self._ema_signal_state = self._alpha * data + (1.0 - self._alpha) * self._ema_signal_state
            
        return self._ema_signal_state
    
    def compute_ema_error_lin_vel_xy(self, data: torch.Tensor) -> torch.Tensor:
        """Update the EMA error state with the new data and return the updated EMA.

        Args:
            data: The current data error (cmd-actual)^2 tensor of shape (num_envs, ...).

        Returns:
            The updated EMA error tensor. If EMA is disabled, returns the original data.
        """
        if not self._enabled:
            return data
        

        if self._ema_error_state_lin_vel_xy is None or self._ema_error_state_lin_vel_xy.shape != data.shape:
            self._init_ema_error_state_lin_vel_xy(data.shape)
            self._ema_error_state_lin_vel_xy.copy_(data)
            return self._ema_error_state_lin_vel_xy

        self._ema_error_state_lin_vel_xy = self._alpha * data + (1.0 - self._alpha) * self._ema_error_state_lin_vel_xy
            
        return self._ema_error_state_lin_vel_xy
    
    def compute_ema_error_ang_vel_z(self, data: torch.Tensor) -> torch.Tensor:
        """Update the EMA error state with the new data and return the updated EMA.

        Args:
            data: The current data error (cmd-actual)^2 tensor of shape (num_envs, ...).

        Returns:
            The updated EMA error tensor. If EMA is disabled, returns the original data.
        """
        if not self._enabled:
            return data
        
        if self._ema_error_state_ang_vel_z is None or self._ema_error_state_ang_vel_z.shape != data.shape:
            self._init_ema_error_state_ang_vel_z(data.shape)
            self._ema_error_state_ang_vel_z.copy_(data)
            return self._ema_error_state_ang_vel_z

        self._ema_error_state_ang_vel_z = self._beta * data + (1.0 - self._beta) * self._ema_error_state_ang_vel_z
            
        return self._ema_error_state_ang_vel_z

    def get_ema_signal(self, data: torch.Tensor) -> torch.Tensor:
        """Return the current EMA signal state.

        Args:
            data: A tensor of the same shape as the data being tracked, used to determine
                  the shape if the EMA signal state has not yet been initialized.

        Returns:
            The current EMA signal tensor. If not initialized, returns a zero tensor of the same shape as `data`.
        """
        if self._ema_signal_state is None:
            return torch.zeros_like(data)
        return self._ema_signal_state

    def reset_ema_signal(self, env_ids: torch.Tensor | None = None, data: torch.Tensor | None = None):
        """Reset the EMA signal state.

        Args:
            env_ids: Environment indices to reset. If None, all environments are reset.
            data: If provided, the EMA signal will be reset to this data for the specified env_ids.
        """
        if self._ema_signal_state is None:
            return

        if env_ids is not None:
            if data is not None:
                # Ensure data is sliced correctly for env_ids
                self._ema_signal_state[env_ids] = data[env_ids]
            else:
                self._ema_signal_state[env_ids] = 0.0
        else:
            if data is not None:
                self._ema_signal_state.copy_(data)
            else:
                self._ema_signal_state.zero_()


    def reset_ema_error(self, env_ids: torch.Tensor | None = None, data: torch.Tensor | None = None):
        """Reset the EMA error state.

        Args:
            env_ids: Environment indices to reset. If None, all environments are reset.
            data: If provided, the EMA error will be reset to this data for the specified env_ids.
        """
        if self._ema_error_state is None:
            return

        if env_ids is not None:
            if data is not None:
                # Ensure data is sliced correctly for env_ids
                self._ema_error_state[env_ids] = data[env_ids]
            else:
                self._ema_error_state[env_ids] = 0.0
        else:
            if data is not None:
                self._ema_error_state.copy_(data)
            else:
                self._ema_error_state.zero_()