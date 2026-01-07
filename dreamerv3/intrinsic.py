"""Intrinsic reward modules for DreamerV3.

This module provides intrinsic reward mechanisms to encourage exploration
and exploitation based on the agent's learning progress and state uncertainty.
"""

import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32


def calc_rew_trend(rewards, decay=0.95):
    """Compute reward EMA and gradient for trend detection.

    Args:
        rewards: (Batch, Time) shaped reward tensor
        decay: Exponential moving average decay rate

    Returns:
        ema: Moving average (same shape as input)
        grad: Gradient/change rate (same shape as input)
    """
    is_vector = rewards.ndim == 2
    if is_vector:
        rewards = rewards[..., None]

    xs = jnp.swapaxes(rewards, 0, 1)  # (Time, Batch, 1)

    def ema_step(carry, x):
        prev_ema = carry
        new_ema = decay * prev_ema + (1.0 - decay) * x
        return new_ema, new_ema

    init = xs[0]
    _, ema_seq = jax.lax.scan(ema_step, init, xs)
    ema = jnp.swapaxes(ema_seq, 0, 1)

    prev_ema = jnp.concatenate([ema[:, :1], ema[:, :-1]], axis=1)
    grad = ema - prev_ema

    if is_vector:
        ema = ema.squeeze(-1)
        grad = grad.squeeze(-1)

    return ema, grad


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


class LexaStyleIntrinsicReward(IntrinsicReward):
    """LEXA-style intrinsic reward with visual curiosity and trend-based weighting.

    Combines:
    1. Reward trend detection using EMA gradient
    2. Visual curiosity from decoder prediction uncertainty (stddev)

    Weighting mechanism:
    - When reward is increasing (rew_grad > 0): weight_upper activates, favoring exploitation
    - When reward is stagnating/decreasing (rew_grad <= 0): weight_lower activates, favoring exploration

    total_rew = weight_upper * ext_rew + weight_lower * visual_bonus

    Reference: agent_update.py implementation, LEXA paper
    """

    enable: bool = False
    decay: float = 0.95
    visual_scale: float = 1.0
    clip_weights: bool = True
    stop_grad_weights: bool = True

    def __init__(self, decoder=None, **kw):
        """Initialize with optional decoder reference.

        Args:
            decoder: Reference to the Decoder module for visual curiosity.
                     If None, falls back to state uncertainty.
            **kw: Other configuration parameters
        """
        super().__init__(**kw)
        self._decoder = decoder

    def set_decoder(self, decoder):
        """Set decoder reference after initialization."""
        self._decoder = decoder

    def __call__(
        self,
        rew_ext: jnp.ndarray,
        imgfeat: dict,
        training: bool = True,
    ) -> tuple:
        metrics = {}

        if not self.enable:
            return rew_ext, metrics

        # 1. Compute reward trend
        rew_ema, rew_grad = calc_rew_trend(rew_ext, decay=self.decay)

        # 2. Compute dynamic weights based on trend
        if self.clip_weights:
            weight_lower = jnp.clip(1 - rew_grad, 0.0, 1.0)  # Fires on stagnation/decrease
            weight_upper = jnp.clip(rew_grad, 0.0, 1.0)      # Fires on increase
        else:
            weight_lower = jax.nn.relu(1 - rew_grad)
            weight_upper = jax.nn.relu(rew_grad)

        # 3. Stop gradient on weights to prevent agent from manipulating them
        if self.stop_grad_weights:
            weight_lower = jax.lax.stop_gradient(weight_lower)
            weight_upper = jax.lax.stop_gradient(weight_upper)

        # 4. Compute visual curiosity bonus
        if self._decoder is not None:
            visual_bonus = self._calc_visual_curiosity(imgfeat)
        else:
            # Fallback: use state uncertainty
            stoch = imgfeat['stoch']
            stoch_flat = stoch.reshape((*stoch.shape[:-2], -1))
            visual_bonus = jnp.std(f32(stoch_flat), axis=-1)

        visual_bonus = visual_bonus * self.visual_scale

        # 5. Combine rewards with trend-based weighting
        total_rew = weight_upper * rew_ext + weight_lower * visual_bonus

        # 6. Record metrics
        metrics['intr/rew_ema_mean'] = rew_ema.mean()
        metrics['intr/rew_grad_mean'] = rew_grad.mean()
        metrics['intr/rew_grad_std'] = rew_grad.std()
        metrics['intr/weight_upper_mean'] = weight_upper.mean()
        metrics['intr/weight_lower_mean'] = weight_lower.mean()
        metrics['intr/visual_bonus_mean'] = visual_bonus.mean()
        metrics['intr/visual_bonus_std'] = visual_bonus.std()
        metrics['intr/r_ext_mean'] = rew_ext.mean()
        metrics['intr/r_total_mean'] = total_rew.mean()

        return total_rew, metrics

    def _calc_visual_curiosity(self, feat):
        """Compute visual curiosity from decoder prediction uncertainty.

        Decodes imagined features to images and uses the prediction
        stddev as an uncertainty measure (requires Normal output from decoder).
        """
        B_K, T = feat['deter'].shape[:2]
        init_carry = self._decoder.initial(B_K)
        reset = jnp.zeros((B_K, T), bool)

        # Decode features - recons should be {key: Agg(Normal(...))}
        _, _, recons = self._decoder(init_carry, feat, reset, training=True)

        def compute_uncertainty(dist):
            # Handle Agg wrapper
            if hasattr(dist, 'output'):
                dist = dist.output

            # Get stddev from Normal distribution
            if hasattr(dist, 'stddev'):
                stddev = dist.stddev
                # Average over spatial dimensions (H, W, C)
                return stddev.mean(axis=(-3, -2, -1))
            else:
                # Fallback for MSE output (no uncertainty available)
                pred = dist.pred() if hasattr(dist, 'pred') else dist
                return jnp.zeros(pred.shape[:-3])

        uncertainties_dict = jax.tree.map(compute_uncertainty, recons)
        uncertainties = jax.tree.leaves(uncertainties_dict)

        if not uncertainties:
            return jnp.zeros((B_K, T), dtype=jnp.float32)

        total_uncertainty = jnp.mean(jnp.stack(uncertainties), axis=0)
        return total_uncertainty


# Registry for factory pattern (extensibility for future intrinsic reward types)
REGISTRY = {
    'adaptive': AdaptiveIntrinsicReward,
    'null': NullIntrinsicReward,
    'lexa_style': LexaStyleIntrinsicReward,
}


def make_intrinsic_reward(config, decoder=None) -> IntrinsicReward:
    """Factory function for creating intrinsic reward modules.

    Args:
        config: Configuration object with 'intrinsic' section containing:
            - enable: bool - master switch for intrinsic rewards
            - typ: str - type of intrinsic reward ('adaptive', 'null', 'lexa_style')
            - <typ>: dict - type-specific parameters
        decoder: Optional decoder reference for visual curiosity (required for lexa_style)

    Returns:
        IntrinsicReward instance configured according to config

    Example config:
        intrinsic:
          enable: True
          typ: lexa_style
          lexa_style:
            decay: 0.95
            visual_scale: 1.0
            clip_weights: True
            stop_grad_weights: True
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
    type_config = dict(config.intrinsic[typ])

    # Pass decoder to types that need it
    if typ == 'lexa_style':
        return cls(decoder=decoder, enable=True, **type_config, name='intrinsic')

    return cls(enable=True, **type_config, name='intrinsic')
