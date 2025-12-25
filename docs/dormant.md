## 休眠ニューロン（dormant neuron）の定義（本論文）

### 記号
- 入力分布: D
- 層 ℓ のニューロン i の活性（activation）: h_i^ℓ(x)（x ∈ D）
- 層 ℓ のニューロン数: H_ℓ

### ニューロンスコア（活動度）の定義
層 ℓ のニューロン i に対して、入力 x を D からサンプルしたときの
「活性の絶対値の平均」を、同一層内で正規化したスコア s_i^ℓ を定義する：

s_i^ℓ
  =  E_{x∈D} |h_i^ℓ(x)|
     --------------------------------------------
      (1/H_ℓ) * Σ_{k=1..H_ℓ} E_{x∈D} |h_k^ℓ(x)|

- 正規化により、同じ層のスコアは総和が 1 になる（層間比較もしやすくする意図）。

### τ-休眠（τ-dormant）の定義
ニューロン i（層 ℓ）が **τ-休眠** であるとは、

s_i^ℓ ≤ τ

を満たすこと。

（直感的には「その層の平均的なニューロン活動に比べて、平均活動が十分小さいニューロン」）

### 関連：休眠ニューロン現象（dormant neuron phenomenon）
学習の進行に伴って **τ-休眠ニューロンの数が継続的に増加** する場合、
そのアルゴリズムは dormant neuron phenomenon を示す、と定義する。


### 実装仕様

#### 実装仕様の要点

- 定義/算出方法: dreamerv3/dormant.py で、層
    内の各特徴の平均絶対活性を計算し、層平均と
    の差で正規化したスコア s_i = mean_abs_i /
    (layer_mean + 1e-8) を用いて s_i <= tau の
    比率を休眠率として算出。
  - サンプル次元: mean_abs_activation は
    bdims（先頭のバッチ次元数）をサンプルとし
    て扱い、残り次元を特徴次元にフラット化。
    agent.report() では bdims=2（B, T）で評
    価。
  - 集約指標: aggregate_metrics は複数層の
    mean_abs を単純連結して休眠率を算出（層ご
    との休眠率の平均ではない）。
  - 無効入力の扱い: None、ndim <= bdims、サン
    プル数 0 の場合はスキップ（該当レイヤはメ
    トリクスに出ない）。

#### 計測対象とメトリクス名

- World Model 系: agent.report() 内で
    outs["tokens"], repfeat["deter"],
    repfeat["stoch"] を対象。
  - Decoder 特徴: dreamerv3/rssm.Decoder に
    return_features=True を追加し、vec/img の
    中間特徴を取得。
  - Reward/Continue Head: embodied/jax/
    heads.py で MLPHead(...,
    return_layers=True) を許可、embodied/jax/
    nets.py の MLP から各隠れ層の出力を取得。
  - Actor/Critic: policy と value の MLP 隠れ
    層出力を同様に取得。
  - メトリクス名: すべて dormant/<name> で出
    力。主な集約は dormant/world_all, dormant/
    world_penultimate, dormant/actor_all,
    レイヤ別は dormant/world_tokens, dormant/
    world_deter, dormant/world_stoch, dormant/
    world_dec_vec, dormant/world_dec_img,
    dormant/world_rew_layer{i}, dormant/
    world_con_layer{i}, dormant/
    actor_layer{i}, dormant/critic_layer{i} な
    ど。

#### 設定/既定値

- dreamerv3/configs.yaml に dormant:
    {enable: False, tau: 0.025} を追加。デフォ
    ルトは無効で report() 時のみ計測。

  テスト

  - embodied/tests/test_dormant.py が休眠率の
    基本計算と集約計算をカバー。

  補足（仕様ドキュメント）

  - docs/dormat.md に論文定義（層内正規化スコ
    アと τ-休眠定義）が明記され、実装はこの定
    義に沿う。


## torch版実装

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch
from torch import Tensor, nn


@dataclass
class _ActivationStat:
    """Accumulator for activation statistics.

    Attributes:
        sum_abs (Tensor): Cumulative absolute activations per feature.
        count (int): Number of samples accumulated.
    """

    sum_abs: Tensor
    count: int = 0


class ActivationRecorder:
    """Utility to record activations and compute dormant ratios.

    Attributes:
        tau (float): Dormant threshold used to classify inactive neurons.
    """

    def __init__(self, tau: float) -> None:
        """Initialize the recorder.

        Args:
            tau (float): Dormant threshold used for later computation.
        """
        self.tau = tau
        self._stats: Dict[str, _ActivationStat] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _update_stat(self, name: str, output: Tensor) -> None:
        """Update the accumulator for a single activation tensor.

        Args:
            name (str): Identifier for the module.
            output (Tensor): Activation tensor produced by the module.
        """
        if output is None or not torch.is_tensor(output):
            return
        with torch.no_grad():
            features = output
            if features.dim() == 0:
                return
            if features.dim() == 1:
                features = features.unsqueeze(0)
            else:
                features = features.flatten(start_dim=1)
            if name not in self._stats:
                self._stats[name] = _ActivationStat(torch.zeros(features.shape[1], device=features.device))
            self._stats[name].sum_abs += features.abs().sum(dim=0)
            self._stats[name].count += features.shape[0]

    def add_module(self, name: str, module: nn.Module | None) -> None:
        """Register a forward hook on the given module.

        Args:
            name (str): Identifier used for the module.
            module (nn.Module | None): Module to hook. Skips if None.
        """
        if module is None:
            return

        def _hook(_: nn.Module, __: Sequence[Tensor] | Tensor, output: Tensor) -> None:
            self._update_stat(name, output)

        self._handles.append(module.register_forward_hook(_hook))

    def close(self) -> None:
        """Remove all registered hooks.

        Returns:
            None
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def compute(self) -> Dict[str, float]:
        """Compute dormant ratios per module and aggregated.

        Returns:
            Dict[str, float]: Mapping from module name to dormant ratio. The special
            key ``all`` aggregates all recorded modules.
        """
        results: Dict[str, float] = {}
        all_means = []
        for name, stat in self._stats.items():
            if stat.count == 0:
                continue
            mean_act = stat.sum_abs / float(stat.count)
            layer_mean = mean_act.mean()
            scores = mean_act / (layer_mean + 1e-8)
            results[name] = (scores <= self.tau).float().mean().item()
            all_means.append(mean_act)
        if all_means:
            stacked = torch.cat(all_means)
            layer_mean = stacked.mean()
            scores = stacked / (layer_mean + 1e-8)
            results["all"] = (scores <= self.tau).float().mean().item()
        return results

    def compute_aggregated(self, keys: Sequence[str], name: str) -> Dict[str, float]:
        """Compute a dormant ratio aggregating selected modules.

        Args:
            keys (Sequence[str]): Module names to aggregate.
            name (str): Name of the resulting aggregated metric.

        Returns:
            Dict[str, float]: A single-entry dict ``{name: ratio}`` if any key had
            data, otherwise an empty dict.
        """
        means = []
        for key in keys:
            stat = self._stats.get(key)
            if stat is None or stat.count == 0:
                continue
            mean_act = stat.sum_abs / float(stat.count)
            means.append(mean_act)
        if not means:
            return {}
        stacked = torch.cat(means)
        layer_mean = stacked.mean()
        scores = stacked / (layer_mean + 1e-8)
        return {name: (scores <= self.tau).float().mean().item()}


def _get_module(module: nn.Module) -> nn.Module:
    """Return the wrapped module if present.

    Args:
        module (nn.Module): Possibly wrapped module.

    Returns:
        nn.Module: The underlying module.
    """
    return getattr(module, "module", module)


@torch.no_grad()
def measure_dormant_neurons_dreamer_v3(
    world_model,
    actor,
    critic,
    batch: Dict[str, Tensor],
    cfg,
    actions_dim: Sequence[int],
    tau: float,
) -> Dict[str, float]:
    """Measure dormant ratios for DreamerV3 world model, actor, and critic.

    Args:
        world_model: The DreamerV3 world model.
        actor: The DreamerV3 actor model.
        critic: The DreamerV3 critic model.
        batch (Dict[str, Tensor]): Batch sampled from replay buffer with shape [T, B, ...].
        cfg: Hydra configuration.
        actions_dim (Sequence[int]): Actions dimension.
        tau (float): Threshold for dormant neurons.

    Returns:
        Dict[str, float]: Dormant ratios keyed by component name.
    """
    batch = {k: v.clone() for k, v in batch.items()}
    device = batch["actions"].device
    batch_size = batch["actions"].shape[1]
    sequence_length = batch["actions"].shape[0]
    stoch_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    stoch_state_size = stoch_size * discrete_size
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size

    wm_recorder = ActivationRecorder(tau)
    actor_recorder = ActivationRecorder(tau)
    critic_recorder = ActivationRecorder(tau)

    encoder = _get_module(world_model.encoder)
    rssm = _get_module(world_model.rssm)
    observation_model = _get_module(world_model.observation_model)
    reward_model = _get_module(world_model.reward_model)
    continue_model = _get_module(world_model.continue_model) if world_model.continue_model else None
    world_penultimate_keys: list[str] = []

    # World model modules
    wm_recorder.add_module("world_encoder_cnn", getattr(encoder, "cnn_encoder", None))
    wm_recorder.add_module("world_encoder_mlp", getattr(encoder, "mlp_encoder", None))
    recurrent_model = _get_module(rssm.recurrent_model)
    wm_recorder.add_module("world_rssm_recurrent_mlp", getattr(recurrent_model, "mlp", None))
    wm_recorder.add_module("world_rssm_recurrent_gru", getattr(recurrent_model, "rnn", None))
    wm_recorder.add_module("world_rssm_transition", _get_module(rssm.transition_model))
    wm_recorder.add_module("world_rssm_representation", _get_module(rssm.representation_model))
    cnn_decoder = getattr(observation_model, "cnn_decoder", None)
    mlp_decoder = getattr(observation_model, "mlp_decoder", None)
    if cnn_decoder is not None:
        wm_recorder.add_module("world_decoder_cnn", _get_module(cnn_decoder).model[0])
        world_penultimate_keys.append("world_decoder_cnn")
    if mlp_decoder is not None:
        wm_recorder.add_module("world_decoder_mlp", _get_module(mlp_decoder).model)
        world_penultimate_keys.append("world_decoder_mlp")
    wm_recorder.add_module("world_reward", _get_module(reward_model).model)
    world_penultimate_keys.append("world_reward")
    if continue_model is not None:
        wm_recorder.add_module("world_continue", _get_module(continue_model).model)
        world_penultimate_keys.append("world_continue")

    # Actor modules
    actor_module = _get_module(actor)
    actor_penultimate_name = "actor_penultimate"
    actor_recorder.add_module(actor_penultimate_name, getattr(actor_module, "model", None))
    actor_linear_modules = (
        [m for m in getattr(actor_module, "model", []).modules() if isinstance(m, nn.Linear)]
        if hasattr(getattr(actor_module, "model", None), "modules")
        else []
    )
    for idx, module in enumerate(actor_linear_modules):
        actor_recorder.add_module(f"actor_layer_{idx}", module)

    # Critic modules
    critic_module = _get_module(critic)
    critic_linear_modules = (
        [m for m in getattr(critic_module, "model", []).modules() if isinstance(m, nn.Linear)]
        if hasattr(getattr(critic_module, "model", None), "modules")
        else []
    )
    critic_linear_names: list[str] = []
    for idx, module in enumerate(critic_linear_modules):
        name = f"critic_layer_{idx}"
        critic_linear_names.append(name)
        critic_recorder.add_module(name, module)
    critic_penultimate_name = critic_linear_names[-2] if len(critic_linear_names) >= 2 else (
        critic_linear_names[-1] if critic_linear_names else None
    )

    batch_obs = {k: batch[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: batch[k] for k in cfg.algo.mlp_keys.encoder})
    batch["is_first"][0, :] = torch.ones_like(batch["is_first"][0, :])
    batch_actions = torch.cat((torch.zeros_like(batch["actions"][:1]), batch["actions"][:-1]), dim=0)

    # Encoder forward pass
    cnn_emb = encoder.cnn_encoder(batch_obs) if encoder.has_cnn_encoder else None
    mlp_emb = encoder.mlp_encoder(batch_obs) if encoder.has_mlp_encoder else None
    if encoder.has_cnn_encoder and encoder.has_mlp_encoder:
        embedded_obs = torch.cat((cnn_emb, mlp_emb), -1)
    elif encoder.has_cnn_encoder:
        embedded_obs = cnn_emb
    else:
        embedded_obs = mlp_emb

    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    if cfg.algo.world_model.decoupled_rssm:
        posteriors_logits, posteriors = rssm._representation(embedded_obs)
        for i in range(sequence_length):
            posterior_input = torch.zeros_like(posteriors[:1]) if i == 0 else posteriors[i - 1 : i]
            recurrent_state, _, _ = rssm.dynamic(
                posterior_input,
                recurrent_state,
                batch_actions[i : i + 1],
                batch["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
    else:
        posteriors = torch.empty(sequence_length, batch_size, stoch_size, discrete_size, device=device)
        posterior = torch.zeros(1, batch_size, stoch_size, discrete_size, device=device)
        for i in range(sequence_length):
            recurrent_state, posterior, _, _, _ = rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                embedded_obs[i : i + 1],
                batch["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            posteriors[i] = posterior

    latent_states = torch.cat((posteriors.view(sequence_length, batch_size, -1), recurrent_states), -1)

    # Decoder, reward, continue
    _ = observation_model(latent_states)
    _ = reward_model(latent_states)
    if continue_model is not None:
        _ = continue_model(latent_states)

    # Actor and critic
    actor_latents = latent_states.view(-1, latent_states.shape[-1])
    _ = actor_module(actor_latents)
    _ = critic_module(actor_latents)

    results: Dict[str, float] = {}
    results.update({k: v for k, v in wm_recorder.compute().items()})
    results.update(wm_recorder.compute_aggregated(world_penultimate_keys, "world_penultimate"))
    actor_res = actor_recorder.compute()
    critic_res = critic_recorder.compute()
    if actor_penultimate_name in actor_res:
        results["actor_penultimate"] = actor_res[actor_penultimate_name]
    if critic_penultimate_name and critic_penultimate_name in critic_res:
        results["critic_penultimate"] = critic_res[critic_penultimate_name]
    if "all" in actor_res:
        results["actor_all"] = actor_res["all"]
    if "all" in critic_res:
        results["critic_all"] = critic_res["all"]
    wm_all = results.pop("all", None)
    if wm_all is not None:
        results["world_all"] = wm_all

    wm_recorder.close()
    actor_recorder.close()
    critic_recorder.close()
    return results
```