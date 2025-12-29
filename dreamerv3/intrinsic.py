"""Intrinsic reward modules for DreamerV3.

This module provides intrinsic reward mechanisms to encourage exploration
and exploitation based on the agent's learning progress and state uncertainty.
"""

import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32


class IntrinsicReward(nj.Module):
    """Base class for intrinsic reward computation.

    Subclasses should implement the __call__ method to compute
    augmented rewards based on extrinsic rewards and state features.
    """

    enable: bool = False

    def __call__(
        self,
        rew_ext: jnp.ndarray,
        imgfeat: dict,
        training: bool = True,
    ) -> tuple:
        """Compute total reward with intrinsic augmentation.

        Args:
            rew_ext: Extrinsic reward predictions, shape (B*K, H+1)
            imgfeat: Imagined features dict with 'stoch', 'deter' keys
            training: Whether in training mode

        Returns:
            rew_total: Combined reward, shape (B*K, H+1)
            metrics: Dict of logged metrics
        """
        raise NotImplementedError


class NullIntrinsicReward(IntrinsicReward):
    """No-op intrinsic reward that passes through extrinsic reward unchanged.

    Used when intrinsic rewards are disabled to minimize overhead.
    """

    enable: bool = False

    def __call__(
        self,
        rew_ext: jnp.ndarray,
        imgfeat: dict,
        training: bool = True,
    ) -> tuple:
        return rew_ext, {}


class AdaptiveIntrinsicReward(IntrinsicReward):
    """Adaptive intrinsic reward using state uncertainty (Std).

    Implements explore/exploit switching based on reward acceleration:
    - r_explor = Std(s')              # Uncertainty-seeking (explore)
    - r_exploit = 1/(Std(s') + eps)   # Certainty-seeking (exploit)

    Gate control:
    - Reward acceleration (improving) -> exploit (reduce exploration)
    - Reward deceleration (stagnating) -> explore (increase exploration)

    The intrinsic reward is adaptively scaled to match the extrinsic
    reward magnitude using an adaptive beta coefficient.

    Reference: docs/intrinsic_reward.md
    """

    enable: bool = False
    beta_max: float = 0.1
    rho: float = 0.1
    epsilon: float = 1e-6
    clip_min: float = -10.0
    clip_max: float = 10.0

    def __call__(
        self,
        rew_ext: jnp.ndarray,
        imgfeat: dict,
        training: bool = True,
    ) -> tuple:
        metrics = {}

        if not self.enable:
            return rew_ext, metrics

        # 1. Compute state uncertainty (Std of stochastic state)
        # stoch shape: (B*K, H+1, stoch, classes) e.g., (*, 32, 32)
        stoch = imgfeat['stoch']
        stoch_flat = stoch.reshape((*stoch.shape[:-2], -1))
        # Compute std across the flattened stochastic dimensions
        std_snext = jnp.std(f32(stoch_flat), axis=-1)
        std_snext = jnp.maximum(std_snext, 0.0)

        # 2. Compute explore/exploit bonuses
        r_explor = std_snext  # Higher uncertainty -> more reward (explore)
        r_exploit = 1.0 / (std_snext + self.epsilon)  # Lower uncertainty -> more reward (exploit)

        # 3. Compute reward velocity (dr) and acceleration (dr - dr_prev)
        # dr_t = r_t - r_{t-1}
        dr = rew_ext[:, 1:] - rew_ext[:, :-1]
        dr_prev = jnp.concatenate(
            [jnp.zeros_like(dr[:, :1]), dr[:, :-1]],
            axis=1
        )

        # 4. Gate control: acceleration -> exploit, deceleration -> explore
        # ReLU(dr - dr_prev): positive when accelerating (reward improving faster)
        # ReLU(dr_prev - dr): positive when decelerating (reward improving slower)
        gate_exploit = jax.nn.relu(dr - dr_prev)
        gate_explor = jax.nn.relu(dr_prev - dr)

        # 5. Combine intrinsic rewards with gating
        # Note: r_explor/r_exploit have shape (B*K, H+1), gates have shape (B*K, H)
        r_hat = gate_exploit * r_exploit[:, 1:] + gate_explor * r_explor[:, 1:]
        r_hat = jnp.concatenate(
            [jnp.zeros_like(r_hat[:, :1]), r_hat],
            axis=1
        )

        # 6. Clip for numerical stability
        r_hat = jnp.clip(r_hat, self.clip_min, self.clip_max)

        # 7. Adaptive beta scaling
        # Scale intrinsic reward to match extrinsic reward magnitude
        ext_scale = jnp.mean(jnp.abs(rew_ext))
        intr_scale = jnp.mean(jnp.abs(r_hat))
        beta = self.rho * ext_scale / (intr_scale + self.epsilon)
        beta = jnp.clip(beta, 0.0, self.beta_max)

        # 8. Combine rewards
        rew_total = rew_ext + beta * r_hat

        # 9. Record metrics for monitoring
        metrics['intr/beta'] = beta
        metrics['intr/r_ext_mean'] = rew_ext.mean()
        metrics['intr/r_hat_mean'] = r_hat.mean()
        metrics['intr/r_hat_std'] = r_hat.std()
        metrics['intr/r_total_mean'] = rew_total.mean()
        metrics['intr/std_snext_mean'] = std_snext.mean()
        metrics['intr/gate_exploit_mean'] = gate_exploit.mean()
        metrics['intr/gate_explor_mean'] = gate_explor.mean()

        return rew_total, metrics


# Registry for factory pattern (extensibility for future intrinsic reward types)
REGISTRY = {
    'adaptive': AdaptiveIntrinsicReward,
    'null': NullIntrinsicReward,
}


def make_intrinsic_reward(config) -> IntrinsicReward:
    """Factory function for creating intrinsic reward modules.

    Args:
        config: Configuration object with 'intrinsic' section containing:
            - enable: bool - master switch for intrinsic rewards
            - typ: str - type of intrinsic reward ('adaptive', 'null')
            - <typ>: dict - type-specific parameters

    Returns:
        IntrinsicReward instance configured according to config

    Example config:
        intrinsic:
          enable: True
          typ: adaptive
          adaptive:
            beta_max: 0.1
            rho: 0.1
            epsilon: 1e-6
            clip_min: -10.0
            clip_max: 10.0
    """
    if not config.intrinsic.enable:
        return NullIntrinsicReward(enable=False, name='intrinsic')

    typ = config.intrinsic.typ
    if typ not in REGISTRY:
        raise ValueError(
            f"Unknown intrinsic reward type: {typ}. "
            f"Available types: {list(REGISTRY.keys())}"
        )

    cls = REGISTRY[typ]
    type_config = config.intrinsic[typ]

    return cls(enable=True, **type_config, name='intrinsic')
