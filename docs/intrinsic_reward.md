# 適応型内発的報酬の実装提案 (標準偏差版)

本ドキュメントでは、学習の進捗（報酬の推移）に応じて探索の度合いを動的に制御する新しい内発的報酬メカニズムの実装について記述します。
不確実性の指標として、分散ではなく**標準偏差（Std）**を採用することで、スケールの安定化を図ります。

## 概要

エージェントの学習効率を向上させるため、報酬の変動（加速度）に基づいた以下の戦略を採用します。

* **報酬が上昇傾向（加速）にある場合**:
    * 現在の行動方針が良いと判断し、**探索を縮小（活用/Exploit）**します。
    * 状態の標準偏差が小さい（確実性が高い）遷移に報酬を与えます。
* **報酬が下降・停滞傾向（減速）にある場合**:
    * 現在の行動方針に行き詰まりがあると判断し、**探索を強化（Explore）**します。
    * 状態の標準偏差が大きい（未知・不確実性が高い）遷移に報酬を与えます。

## 数式定義

時点 $t$ における外発的報酬を $r_t$、次状態の標準偏差を $Std(s')$ とした際、内発的報酬 $\hat{r}_t$ は以下のように定義されます。

$$
\begin{aligned}
r_{explor} &= Std(s') \\
r_{exploit} &= \frac{1}{Std(s') + \epsilon} \\
\Delta r_t &= r_t - r_{t-1} \\
\hat{r}_t &= \text{ReLU}(\Delta r_t - \Delta r_{t-1}) \cdot r_{exploit} + \text{ReLU}(\Delta r_{t-1} - \Delta r_t) \cdot r_{explor}
\end{aligned}
$$

最終的な報酬 $r_{total}$ は、適応係数 $\beta$ を用いて以下のように計算されます。

$$
r_{total} = r_{extrinsic} + \beta \cdot \hat{r}_t
$$

---

## 実装比較

### 1. 変更前（旧実装）

外発的報酬予測値をそのまま損失関数へ入力しています。

```python
inp = self.feat2tensor(imgfeat)
los, imgloss_out, mets = imag_loss(
    imgact,
    self.rew(inp, 2).pred(),  # 外発的報酬のみを使用
    self.con(inp, 2).prob(1),
    self.pol(inp, 2),
    self.val(inp, 2),
    self.slowval(inp, 2),
    self.retnorm, self.valnorm, self.advnorm,
    update=training,
    contdisc=self.config.contdisc,
    horizon=self.config.horizon,
    **self.config.imag_loss)
```

### 2. 変更後（新実装：標準偏差を使用）
分散（var）ではなく標準偏差（std）を用いて不確実性を計算します。

```python
inp = self.feat2tensor(imgfeat)

# 外発的報酬の予測
rew_ext = self.rew(inp, 2).pred()

# --- 内発的報酬の計算開始 ---
eps = 1e-6

# 次状態の標準偏差計算 (確率的表現の不確実性を取得)
try:
    stoch = imgfeat['stoch']
    stoch_flat = stoch.reshape((*stoch.shape[:-2], -1))
    # 分散(var)ではなく標準偏差(std)を使用
    std_snext = jnp.std(f32(stoch_flat), axis=-1)
except Exception:
    # フォールバック
    std_snext = jnp.std(f32(inp), axis=-1)

std_snext = jnp.maximum(std_snext, 0.0)

# 探索ボーナス (stdが大きいほど報酬) と 活用ボーナス (stdが小さいほど報酬)
r_explor = std_snext
r_exploit = 1.0 / (std_snext + eps)

# 報酬の変化量(速度)と、変化量の変化(加速度)の計算
dr = rew_ext[:, 1:] - rew_ext[:, :-1]
dr_prev = jnp.concatenate(
    [jnp.zeros_like(dr[:, :1]), dr[:, :-1]],
    axis=1
)

# ゲート制御: 加速時は活用、減速時は探索
gate_exploit = jax.nn.relu(dr - dr_prev)
gate_explor = jax.nn.relu(dr_prev - dr)

# 内発的報酬の合成
r_hat = gate_exploit * r_exploit[:, 1:] + gate_explor * r_explor[:, 1:]
r_hat = jnp.concatenate(
    [jnp.zeros_like(r_hat[:, :1]), r_hat],
    axis=1
)

# 安定化のためのクリッピング
r_hat = jnp.clip(r_hat, -10.0, 10.0)

# 適応的な係数 beta の計算
beta_max = 0.1
rho = 0.1

ext_scale = jnp.mean(jnp.abs(rew_ext))
intr_scale = jnp.mean(jnp.abs(r_hat))

# 外発的報酬と内発的報酬のスケールを合わせる
beta = rho * ext_scale / (intr_scale + eps)
beta = jnp.clip(beta, 0.0, beta_max)

# 最終報酬の算出
rew_total = rew_ext + beta * r_hat

# メトリクスの記録
metrics['intr/beta'] = beta
metrics['intr/r_ext_mean'] = rew_ext.mean()
metrics['intr/r_hat_mean'] = r_hat.mean()
metrics['intr/r_total_mean'] = rew_total.mean()
# --- 計算終了 ---

los, imgloss_out, mets = imag_loss(
    imgact,
    rew_total,  # 合成した報酬を使用
    self.con(inp, 2).prob(1),
    self.pol(inp, 2),
    self.val(inp, 2),
    self.slowval(inp, 2),
    self.retnorm, self.valnorm, self.advnorm,
    update=training,
    contdisc=self.config.contdisc,
    horizon=self.config.horizon,
    **self.config.imag_loss)
```
