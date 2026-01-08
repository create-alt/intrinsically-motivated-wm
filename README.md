# Intrinsically Motivated World Model

A fork of [DreamerV3](https://github.com/danijar/dreamerv3) with exploration enhancements.

DreamerV3のフォークに様々な「内発的報酬」を追加した実装です。

---

## Overview / 概要

[DreamerV3](https://arxiv.org/pdf/2301.04104) learns a world model from experiences and uses it to train an actor-critic policy from imagined trajectories. This fork extends DreamerV3 with intrinsic motivation and advanced replay strategies for improved exploration.

DreamerV3は経験から世界モデルを学習し、想像された軌道からActor-Criticポリシーを訓練します。このフォークでは、**「内発的報酬」** と **「リプレイバッファからのサンプリング方法の工夫（広義の内発的報酬）」** を追加し、性能の向上を目指したものです。

---

## Added Features / 追加機能

This implementation adds three main categories of features:

本実装では、以下の3つのカテゴリの機能を追加しています：

1. **Intrinsic Reward / 内発的報酬** - Reward shaping applied during imagination rollouts / 想像ロールアウト時に適用される報酬形成
2. **Replay Sampling Strategies / リプレイサンプリング戦略** - Priority-based experience sampling (broadly interpreted as intrinsic motivation) / 優先度ベースの経験サンプリング（広義の内発的報酬）
3. **Dormant Neuron Monitoring / 休眠ニューロン計測** - Network health diagnostics / ネットワーク健全性診断

---

### 1. Intrinsic Reward / 内発的報酬

Intrinsic rewards are added to the extrinsic reward during policy learning in imagined trajectories. This encourages exploration by providing bonus rewards for visiting uncertain or novel states.

内発的報酬は、想像された軌道でのポリシー学習時に外発的報酬に加算されます。不確実または新規な状態への訪問にボーナス報酬を与えることで、探索を促進します。

#### 1.1 Adaptive Intrinsic Reward

Balances exploration and exploitation based on reward acceleration (second derivative of reward). When rewards are accelerating (improving), exploitation is favored; when rewards are stagnating, exploration is encouraged.

報酬の加速度（報酬の2階微分）に基づいて探索と活用をバランシングします。報酬が加速している（改善している）ときは活用を優先し、報酬が停滞しているときは探索を促進します。

- **Exploration bonus**: `r_explore = Std(s')` (state uncertainty)
- **Exploitation bonus**: `r_exploit = 1 / (Std(s') + ε)` (state certainty)
- **Adaptive scaling**: Automatically scales intrinsic reward relative to extrinsic reward magnitude

```yaml
agent:
  intrinsic:
    enable: True
    typ: adaptive
    adaptive:
      beta_max: 0.1    # Maximum intrinsic reward scale
      rho: 0.1         # Adaptive scaling coefficient
      epsilon: 1e-6    # Numerical stability
      clip_min: -10.0  # Reward clipping
      clip_max: 10.0
```

#### 1.2 LEXA-style Intrinsic Reward

Uses decoder prediction uncertainty as visual curiosity signal, combined with reward trend detection via exponential moving average (EMA).

デコーダの予測不確実性を視覚的好奇心シグナルとして使用し、指数移動平均（EMA）による報酬トレンド検出と組み合わせます。

- Uses decoder's Normal distribution stddev as uncertainty measure
- EMA-based reward trend detection for dynamic weighting

```yaml
agent:
  intrinsic:
    enable: True
    typ: lexa_style
    lexa_style:
      decay: 0.95           # EMA decay rate
      visual_scale: 1.0     # Visual curiosity scaling
      clip_weights: True
      stop_grad_weights: True
```

---

### 2. Replay Sampling Strategies / リプレイサンプリング戦略

These strategies modify how experiences are sampled from the replay buffer, prioritizing certain transitions to improve learning efficiency. This can be viewed as a form of intrinsic motivation at the data selection level.

これらの戦略は、リプレイバッファからの経験のサンプリング方法を変更し、学習効率を向上させるために特定の遷移を優先します。これはデータ選択レベルでの内発的動機付けの一形態と見なせます。

#### 2.1 Curious Replay

Based on [Curious Replay for Model-based Adaptation](https://arxiv.org/abs/2306.15934) (Kauvar et al., ICML 2023). Implementation reference: [cr-dv3](https://github.com/AutonomousAgentsLab/cr-dv3).

Combines count-based novelty with model prediction error:

カウントベースの新規性とモデル予測誤差を組み合わせます：

```
priority = c × β^visit_count + (model_loss + ε)^α
```

- **Count-based term**: `c × β^visit_count` - Prioritizes less-visited experiences / 訪問回数が少ない経験を優先
- **Loss-based term**: `(model_loss + ε)^α` - Prioritizes hard-to-predict transitions / 予測が困難な遷移を優先

**Entropy adjustment (extension):** This implementation adds an optional entropy-based adjustment to the loss term:

**エントロピー調整（拡張）:** 本実装では、損失項にオプションのエントロピーベース調整を追加しています：

```
adjusted_loss = max(model_loss - λ × entropy(stoch), 0)
```

When `entropy_lambda = 0`, the behavior matches the original Curious Replay paper exactly. Setting `λ > 0` reduces priority for transitions where high loss is due to inherent stochasticity rather than model uncertainty.

`entropy_lambda = 0` のとき、元の Curious Replay 論文と完全に一致します。`λ > 0` に設定すると、高い損失がモデルの不確実性ではなく本質的な確率性に起因する遷移の優先度を下げます。

```yaml
replay:
  fracs: {curious: 1.0, uniform: 0.0}
  curious:
    c: 1e4           # Count-based coefficient
    beta: 0.7        # Visit count decay
    alpha: 0.7       # Loss term exponent
    epsilon: 0.01    # Numerical stability
    initial: inf     # Initial priority for new items
    entropy_lambda: 0.0  # 0 = original paper, >0 = entropy-adjusted
```

Or use the preset / プリセットを使用:
```sh
--configs atari curious_replay
```

#### 2.2 Explore/Exploit Balancing (TrendMixture)

Dynamically adjusts the ratio between exploration-focused and exploitation-focused sampling based on reward trends.

報酬トレンドに基づいて、探索重視と活用重視のサンプリング比率を動的に調整します。

- **Explore priority**: `KL(posterior || prior)` - High uncertainty → exploration / 高い不確実性 → 探索
- **Exploit priority**: `1 / KL` - Low uncertainty → exploitation / 低い不確実性 → 活用
- **TrendMixture**: Uses fast/slow EMA to detect reward trends and adjust gate / 速い/遅いEMAで報酬トレンドを検出しゲートを調整

```yaml
replay:
  fracs: {explore: 0.5, exploit: 0.5, uniform: 0.0}
  trend:
    enable: True
    fast: 0.1        # Fast EMA decay
    slow: 0.01       # Slow EMA decay
    k: 5.0           # Trend sensitivity
    eps: 1e-6
    gate_min: 0.05   # Minimum gate value
    gate_max: 0.95   # Maximum gate value
    gate_init: 0.5   # Initial gate (0=explore, 1=exploit)
```

---

### 3. Dormant Neuron Monitoring / 休眠ニューロン計測

Based on [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://arxiv.org/abs/2302.12902) (Sokar et al., ICML 2023).

Monitors the health of neural network layers by tracking dormant neurons - neurons with very low activation relative to the layer average.

休眠ニューロン（層平均に対して非常に低い活性化を持つニューロン）を追跡することで、ニューラルネットワーク層の健全性を監視します。

**Definition / 定義:**
- Neuron score: `s_i = mean_abs_activation_i / layer_mean`
- Dormant if: `s_i ≤ τ` (default τ = 0.025)
- Dormant ratio: Fraction of dormant neurons in a layer

**Monitored components / 監視対象:**
- World model: tokens, deterministic state, stochastic state, decoder features
- Reward and continuation heads (per layer)
- Actor network (per layer)
- Critic network (per layer)

```yaml
agent:
  dormant:
    enable: True
    tau: 0.025  # Dormancy threshold
```

**Reported metrics / レポートされるメトリクス:**
- `dormant/world_tokens`, `dormant/world_deter`, `dormant/world_stoch`
- `dormant/world_all`, `dormant/world_penultimate`
- `dormant/actor_all`, `dormant/critic_all`
- `dormant/world_rew_layer*`, `dormant/actor_layer*`, etc.

---

## Installation / インストール

Requires Python 3.11+. Install with [UV](https://github.com/astral-sh/uv) (recommended):

```sh
uv venv
source .venv/bin/activate
uv sync
```

---

## Quick Start / クイックスタート

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari \
  --task atari_ms_pacman
```

With Curious Replay / Curious Replayを使用:
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari curious_replay \
  --task atari_ms_pacman
```

---

## License / ライセンス

MIT License - Copyright (c) 2024 Danijar Hafner

This fork maintains the original license. See [LICENSE](LICENSE) for details.

---

## Citation / 引用

If you use this code, please cite the original papers:

**DreamerV3:**
```bibtex
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

**Curious Replay:**
```bibtex
@inproceedings{kauvar2023curious,
  title={Curious Replay for Model-based Adaptation},
  author={Kauvar, Isaac and Doyle, Chris and Zhou, Linqi and Haber, Nick},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

**Dormant Neuron:**
```bibtex
@inproceedings{sokar2023dormant,
  title={The Dormant Neuron Phenomenon in Deep Reinforcement Learning},
  author={Sokar, Ghada and Agarwal, Rishabh and Castro, Pablo Samuel and Evci, Utku},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## Links

- [Original DreamerV3 Repository](https://github.com/danijar/dreamerv3)
- [DreamerV3 Paper](https://arxiv.org/pdf/2301.04104)
- [DreamerV3 Project Website](https://danijar.com/dreamerv3)
- [Curious Replay Repository (cr-dv3)](https://github.com/AutonomousAgentsLab/cr-dv3)
- [Curious Replay Paper](https://arxiv.org/abs/2306.15934)
- [Dormant Neuron Paper](https://arxiv.org/abs/2302.12902)
