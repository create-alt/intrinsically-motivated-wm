# 内発的動機付け世界モデル

DreamerV3のフォークに対して「内発的報酬」の機能を追加したリポジトリである。

本リポジトリは DreamerV3 をフォークし、様々な「内発的報酬」の機能を追加した実装である。

---

## 概要

[DreamerV3](https://arxiv.org/pdf/2301.04104) は経験から世界モデルを学習し、想像上の軌道を用いて Actor-Critic を訓練するモデルである。このフォークでは、**「内発的報酬」** および **「リプレイバッファからのサンプリング戦略（広義の内発的報酬）」** を追加することで、探索性能の向上を目指している。

---

## 実験結果

実験結果の詳細は [RESULTS.md](RESULTS.md) を参照されたい。

---

## 追加機能

本実装では、主に以下3つの機能カテゴリを追加している：

1. **Intrinsic Reward / 内発的報酬** - 想像ロールアウト時に適用される報酬形成
2. **Replay Sampling Strategies / リプレイサンプリング戦略** - 優先度ベースの経験サンプリング（広義の内発的報酬）
3. **Dormant Neuron Monitoring / 休眠ニューロン計測** - ネットワーク健全性診断

---

### 1. 内発的報酬

内発的報酬は、想像上の軌道におけるポリシー学習の際、外発的報酬に加算される。不確実性が高い状態や新規の状態への訪問に対してボーナスを与えることで、探索を促進する。

#### 1.1 適応的内発的報酬 (Adaptive Intrinsic Reward)

報酬の加速度（2階微分）に基づき、探索と活用のバランスを調整する。報酬が増加傾向にある（加速している）場合は活用を優先し、停滞している場合は探索を促進する。

- **Exploration bonus**: `r_explore = Std(s')` (state uncertainty / 状態の不確実性)
- **Exploitation bonus**: `r_exploit = 1 / (Std(s') + ε)` (state certainty / 状態の確実性)
- **Adaptive scaling**: Automatically scales intrinsic reward relative to extrinsic reward magnitude / 外発的報酬の大きさに応じて内発的報酬を自動スケーリング

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

#### 1.2 LEXA型内発的報酬 (LEXA-style Intrinsic Reward)

デコーダの予測不確実性を視覚的な好奇心シグナルとして利用し、指数移動平均（EMA）を用いた報酬トレンド検出と組み合わせる。

- デコーダの正規分布の標準偏差を不確実性尺度として使用
- EMAベースの報酬トレンド検出による動的な重み付け

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

### 2. リプレイサンプリング戦略



この戦略では、リプレイバッファからのサンプリング方法を変更し、特定の遷移を優先的に抽出することで学習効率の向上を図る。これはデータ選択レベルにおける内発的動機付けの一形態とみなせる。

#### 2.1 Curious Replay（サンプリング戦略のベースライン）

[Curious Replay for Model-based Adaptation](https://arxiv.org/abs/2306.15934) (Kauvar et al., ICML 2023) に基づく。実装の参考: [cr-dv3](https://github.com/AutonomousAgentsLab/cr-dv3)。

カウントベースの新規性とモデル予測誤差を組み合わせた手法である。

```
priority = c × β^visit_count + (model_loss + ε)^α
```

- **Count-based term**: `c × β^visit_count` - 訪問頻度の低い経験を優先
- **Loss-based term**: `(model_loss + ε)^α` - 予測困難な遷移を優先



**エントロピー調整（拡張機能）:** 本実装では、損失項に対してエントロピーベースの調整をオプションとして追加している：

```
adjusted_loss = max(model_loss - λ × entropy(stoch), 0)
```



`entropy_lambda = 0` の場合、オリジナルの Curious Replay 論文と完全に一致する挙動となる。`λ > 0` に設定すると、モデルの不確実性ではなく環境の確率的性質（ノイズ）に起因して損失が高くなる遷移の優先度を下げることができる。

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

プリセットの使用例：
```sh
--configs atari curious_replay
```

#### 2.2 探索/活用バランシング (TrendMixture)



報酬のトレンドに基づき、探索重視のサンプリングと活用重視のサンプリングの比率を動的に調整する。

- **Explore priority**: `KL(posterior || prior)` - 不確実性が高い → とりあえず探索
- **Exploit priority**: `1 / KL` - 不確実性が低い（自信がある） → 知識を活用
- **TrendMixture**: スパンの短いEMAと長いEMAを使って報酬のトレンドを検知し、ゲートを調整する

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

### 3. 休眠ニューロン計測

[The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://arxiv.org/abs/2302.12902) (Sokar et al., ICML 2023) に基づく。



休眠ニューロン（層全体の平均に対し、活性化レベルが著しく低いニューロン）を追跡し、ニューラルネットワークの健全性を監視する。

**定義:**
- Neuron score: `s_i = mean_abs_activation_i / layer_mean`
- Dormant if: `s_i ≤ τ` (default τ = 0.025)
- Dormant ratio: Fraction of dormant neurons in a layer

**監視対象:**
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

**記録されるメトリクス:**
- `dormant/world_tokens`, `dormant/world_deter`, `dormant/world_stoch`
- `dormant/world_all`, `dormant/world_penultimate`
- `dormant/actor_all`, `dormant/critic_all`
- `dormant/world_rew_layer*`, `dormant/actor_layer*`, etc.

---

## インストール

Python 3.11+ が必要である。[UV](https://github.com/astral-sh/uv) を用いたインストールを推奨する:

```sh
uv venv
source .venv/bin/activate
uv sync
```

---

## クイックスタート

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari \
  --task atari_ms_pacman
```

Curious Replay を使う場合:
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs atari curious_replay \
  --task atari_ms_pacman
```

---

## ライセンス

MIT License - Copyright (c) 2024 Danijar Hafner

This fork maintains the original license. See [LICENSE](LICENSE) for details. (本フォークはオリジナルのライセンスを維持している。詳細は [LICENSE](LICENSE) を参照されたい。)

---

## 引用

このコードを使用する場合は、原著論文を引用されたい:

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

## リンク

- [Original DreamerV3 Repository](https://github.com/danijar/dreamerv3)
- [DreamerV3 Paper](https://arxiv.org/pdf/2301.04104)
- [DreamerV3 Project Website](https://danijar.com/dreamerv3)
- [Curious Replay Repository (cr-dv3)](https://github.com/AutonomousAgentsLab/cr-dv3)
- [Curious Replay Paper](https://arxiv.org/abs/2306.15934)
- [Dormant Neuron Paper](https://arxiv.org/abs/2302.12902)

---
