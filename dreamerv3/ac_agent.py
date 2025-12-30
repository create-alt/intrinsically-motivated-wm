import re

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

import embodied.jax
import embodied.jax.nets as nn

from . import dormant, rssm
from .agent import lambda_return

f32 = jnp.float32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)


class Agent(embodied.jax.Agent):
    """Actor-critic agent without a world model."""

    banner = [
        r"---   _   ___   ___  ___   ___  ___   ___  ---",
        r"---  /_\ / __| / __|/ _ \ / __|/ _ \ / __| ---",
        r"--- / _ \ (__ | (__| (_) | (__| (_) | (__  ---",
        r"---/_/ \_\___| \___|\___/ \___|\___/ \___| ---",
    ]

    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config

        exclude = ("is_first", "is_last", "is_terminal", "reward")
        enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
        self.enc = {
            "simple": rssm.Encoder,
        }[
            config.enc.typ
        ](enc_space, **config.enc[config.enc.typ], name="enc")

        self.feat_dim = self._feat_dim(config)
        self.feat = nn.Linear(self.feat_dim, name="feat")

        d1, d2 = config.policy_dist_disc, config.policy_dist_cont
        outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
        self.pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name="pol")

        scalar = elements.Space(np.float32, ())
        self.val = embodied.jax.MLPHead(scalar, **config.value, name="val")
        self.slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="slowval"),
            source=self.val,
            **config.slowvalue,
        )

        self.retnorm = embodied.jax.Normalize(**config.retnorm, name="retnorm")
        self.valnorm = embodied.jax.Normalize(**config.valnorm, name="valnorm")
        self.advnorm = embodied.jax.Normalize(**config.advnorm, name="advnorm")

        self.modules = [
            self.enc,
            self.feat,
            self.pol,
            self.val,
        ]
        self.opt = embodied.jax.Optimizer(
            self.modules, self._make_opt(**config.opt), summary_depth=1, name="opt"
        )

        self.scales = {k: config.loss_scales[k] for k in ("policy", "value")}

    @property
    def policy_keys(self):
        return "^(enc|feat|pol)/"

    @property
    def ext_space(self):
        spaces = {}
        spaces["consec"] = elements.Space(np.int32)
        spaces["stepid"] = elements.Space(np.uint8, 20)
        return spaces

    def init_policy(self, batch_size):
        return {}

    def init_train(self, batch_size):
        return {}

    def init_report(self, batch_size):
        return {}

    def policy(self, carry, obs, mode="train"):
        reset = obs["is_first"]
        enc_carry, _, tokens = self.enc(carry, obs, reset, training=False, single=True)
        feat = self.feat(tokens)
        policy = self.pol(feat, bdims=1)
        act = sample(policy)
        out = {}
        out["finite"] = elements.tree.flatdict(
            jax.tree.map(
                lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
                dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
            )
        )
        return enc_carry, act, out

    def train(self, carry, data):
        carry, obs, act, _ = self._apply_replay_context(carry, data)
        metrics, (carry, outs, mets) = self.opt(
            self.loss, carry, obs, act, training=True, has_aux=True
        )
        metrics.update(mets)
        self.slowval.update()
        return carry, outs, metrics

    def loss(self, carry, obs, act, training, return_layers=False):
        reset = obs["is_first"]
        carry, _, tokens = self.enc(carry, obs, reset, training)
        feat = self.feat(tokens)
        if return_layers:
            policy, pol_layers = self.pol(feat, 2, return_layers=True)
            value, val_layers = self.val(feat, 2, return_layers=True)
        else:
            policy = self.pol(feat, 2)
            value = self.val(feat, 2)
            pol_layers, val_layers = None, None
        slowvalue = self.slowval(feat, 2)

        losses, outs, metrics = ac_loss(
            act,
            obs["is_last"],
            obs["is_terminal"],
            obs["reward"],
            policy,
            value,
            slowvalue,
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )

        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()),
            sorted(self.scales.keys()),
        )
        metrics.update({f"loss/{k}": v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        outs["losses"] = losses
        if return_layers:
            outs["pol_layers"] = pol_layers
            outs["val_layers"] = val_layers
        return loss, (carry, outs, metrics)

    def report(self, carry, data):
        if not self.config.report:
            return carry, {}

        carry, obs, act, _ = self._apply_replay_context(carry, data)
        metrics = {}
        _, (new_carry, outs, mets) = self.loss(
            carry, obs, act, training=False, return_layers=self.config.dormant.enable
        )
        metrics.update(mets)

        if self.config.dormant.enable:
            tau = self.config.dormant.tau
            bdims = 2
            actor_means = []
            critic_means = []

            def add_metric(name, tensor, target_means):
                mean_abs = dormant.mean_abs_activation(tensor, bdims)
                if mean_abs is None:
                    return
                metrics[f"dormant/{name}"] = dormant.dormant_ratio(mean_abs, tau)
                target_means.append(mean_abs)

            for idx, layer in enumerate(outs.get("pol_layers", []) or []):
                add_metric(f"actor_layer{idx}", layer, actor_means)
            for idx, layer in enumerate(outs.get("val_layers", []) or []):
                add_metric(f"critic_layer{idx}", layer, critic_means)

            actor_all = dormant.aggregate_metrics(actor_means, tau)
            if actor_all is not None:
                metrics["dormant/actor_all"] = actor_all
            critic_all = dormant.aggregate_metrics(critic_means, tau)
            if critic_all is not None:
                metrics["dormant/critic_all"] = critic_all

        return new_carry, metrics

    def _apply_replay_context(self, carry, data):
        stepid = data["stepid"]
        obs = {k: data[k] for k in self.obs_space}
        act = {k: data[k] for k in self.act_space}
        if not self.config.replay_context:
            return carry, obs, act, stepid
        k = self.config.replay_context
        rhs = lambda xs: jax.tree.map(lambda x: x[:, k:], xs)
        obs = rhs(obs)
        act = rhs(act)
        stepid = rhs(stepid)
        return carry, obs, act, stepid

    def _feat_dim(self, config) -> int:
        """Compute Dreamer feature dimension for actor/critic alignment.

        Args:
            config: Agent config containing RSSM settings.

        Returns:
            int: Feature dimension for actor/critic inputs.

        Raises:
            ValueError: If the RSSM config is missing required fields.
        """
        try:
            rssm = config.dyn[config.dyn.typ]
            return int(rssm.deter + rssm.stoch * rssm.classes)
        except Exception as exc:
            raise ValueError("RSSM config missing for feature sizing.") from exc

    def _make_opt(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r"/kernel$",
        schedule: str = "const",
        warmup: int = 1000,
        anneal: int = 0,
    ):
        chain = []
        chain.append(embodied.jax.opt.clip_by_agc(agc))
        chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
        chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
        if wd:
            assert not wdregex[0].isnumeric(), wdregex
            pattern = re.compile(wdregex)
            wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
            chain.append(optax.add_decayed_weights(wd, wdmask))
        assert anneal > 0 or schedule == "const"
        if schedule == "const":
            sched = optax.constant_schedule(lr)
        elif schedule == "linear":
            sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == "cosine":
            sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
        else:
            raise NotImplementedError(schedule)
        if warmup:
            ramp = optax.linear_schedule(0.0, lr, warmup)
            sched = optax.join_schedules([ramp, sched], [warmup])
        chain.append(optax.scale_by_learning_rate(sched))
        return optax.chain(*chain)


def ac_loss(
    act,
    last,
    term,
    rew,
    policy,
    value,
    slowvalue,
    retnorm,
    valnorm,
    advnorm,
    update,
    contdisc=True,
    slowtar=False,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    """Compute actor-critic losses on real trajectories.

    Args:
        act: Action dict from replay, shape (B, T, ...).
        last: Episode boundary indicator, shape (B, T).
        term: Terminal indicator, shape (B, T).
        rew: Reward array, shape (B, T).
        policy: Policy distribution outputs.
        value: Value distribution outputs.
        slowvalue: Target value distribution outputs.
        retnorm: Return normalizer.
        valnorm: Value normalizer.
        advnorm: Advantage normalizer.
        update: Whether to update normalizers.
        contdisc: Whether to use continuous discounting.
        slowtar: Whether to use slow value targets.
        horizon: Planning horizon for discounting.
        lam: Lambda for TD(lambda).
        actent: Action entropy regularization coefficient.
        slowreg: Slow value regularization coefficient.

    Returns:
        tuple: (losses, outs, metrics).
    """
    losses = {}
    metrics = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 if contdisc else 1 - 1 / horizon
    con = f32(~term)
    weight = jnp.cumprod(disc * con, 1) / disc
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale
    logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
    ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
    policy_loss = sg(weight[:, :-1]) * -(
        logpi * sg(adv_normed) + actent * sum(ents.values())
    )
    losses["policy"] = policy_loss

    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    value_loss = (
        value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred()))
    )[:, :-1]
    losses["value"] = sg(weight[:, :-1]) * value_loss

    ret_normed = (ret - roffset) / rscale
    metrics["adv"] = adv.mean()
    metrics["adv_std"] = adv.std()
    metrics["adv_mag"] = jnp.abs(adv).mean()
    metrics["rew"] = rew.mean()
    metrics["ret"] = ret_normed.mean()
    metrics["val"] = val.mean()
    metrics["tar"] = tar_normed.mean()
    metrics["weight"] = weight.mean()
    metrics["slowval"] = slowval.mean()
    metrics["ret_min"] = ret_normed.min()
    metrics["ret_max"] = ret_normed.max()
    metrics["ret_rate"] = (jnp.abs(ret_normed) >= 1.0).mean()
    for k in act:
        metrics[f"ent/{k}"] = ents[k].mean()
        if hasattr(policy[k], "minent"):
            lo, hi = policy[k].minent, policy[k].maxent
            metrics[f"rand/{k}"] = (ents[k].mean() - lo) / (hi - lo)

    outs = {}
    outs["ret"] = ret
    return losses, outs, metrics
