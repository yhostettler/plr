from dataclasses import dataclass
from isaaclab.utils import configclass


@configclass
class EMAManagerCfg:
    """Configuration for the EMA (Exponential Moving Average) manager.

    This manager maintains an exponential moving average of a linear velocity xy and angular z velocity
    to provide a smoothed version for observations or rewards.
    """

    enabled: bool = True
    """Whether to enable EMA tracking."""

    alpha: float = 0.1
    """Smoothing factor for the linear velocity xy EMA. 
    EMA_new = alpha * current_value + (1 - alpha) * EMA_old.
    A smaller alpha means more smoothing (more weight on history).
    """

    beta: float = 0.1
    """Smoothing factor for the angualr velocity z EMA. 
    EMA_new = beta * current_value + (1 - beta) * EMA_old.
    A smaller beata means more smoothing (more weight on history).
    """
