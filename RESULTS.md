# 実験結果

Ms. Pacmanにおける実験結果のまとめである。実験に使用したスクリプトは `exp_script` ディレクトリに配置されている。

---

## 1. ベースライン実験の再現

**スクリプト:**
- [`exp_script/251226_run_mspackman_dormat.sh`](exp_script/251226_run_mspackman_dormat.sh)

### 仮説
- 論文のハイパーパラメータを用いてDreamerV3実験を再現する

### 結果

#### 報酬

- 論文で提示されている性能と概ね一致した

![Baseline Result](./assets/mspackman_baseline.png)

#### 休眠ニューロン計測

- 今回の全ての実験では、休眠ニューロンの割合を計測し、学習の進捗やネットワークの健全性の目安とした。

![Dormant Neurons](./assets/mspackman_dormant.png)

### 考察

- 論文実装の再現ができた
- 休眠ニューロンの割合は低く保たれており、これが世界モデルの強みである可能性がある


---

## 2. 内発的報酬

### 2.1 適応的内発的報酬 (Adaptive Intrinsic Reward)

**スクリプト:**
- [`exp_script/251229_run_mspackman_intrinsic.sh`](exp_script/251229_run_mspackman_intrinsic.sh)

#### 仮説
- 報酬が上がっているときは、活用を促進し、報酬が下がっているときは探索を促進するような内発的報酬を設計
- 探索と活用を必要に応じて切り替えるような方策の獲得を期待した

#### 結果
- 明確な改善は見られなかった

![Adaptive Intrinsic Reward](./assets/mspackman_adaptive.png)

#### 考察
- 実装について、H(stoch)が必ずしも、
- 報酬の上昇・下降トレンドが頻繁に変化し、報酬が不安定になった可能性

### 2.2 LEXA型内発的報酬 (LEXA-style Intrinsic Reward)

**スクリプト:**
- [`exp_script/260107_run_mspackman_normal.sh`](exp_script/260107_run_mspackman_normal.sh)
- [`exp_script/260108_run_mspackman_normal_intrinsic.sh`](exp_script/260108_run_mspackman_normal_intrinsic.sh)

#### 仮説
<!-- Describe the hypothesis here / ここに仮説を記述してください -->

#### 結果

![LEXA-style Intrinsic Reward](./assets/mspackman_lexa_style.png)

#### 考察
<!-- Describe the discussion here / ここに考察を記述してください -->

---

## 3. リプレイサンプリング戦略

これらの手法は学習データの分布を変化させるため、広義の内発的動機付けとみなすことができる。

### 3.1 Curious Replay

**スクリプト:**
- [`exp_script/251229_run_mspackman_curious.sh`](exp_script/251229_run_mspackman_curious.sh)

#### 仮説
<!-- Describe the hypothesis here / ここに仮説を記述してください -->

#### 結果

![Curious Replay](./assets/mspackman_curious.png)

#### 考察
<!-- Describe the discussion here / ここに考察を記述してください -->

### 3.2 Curious Replay (エントロピー正則化)

Comparing Curious Replay with entropy regularization enabled (`H(stoch)`).

エントロピー正則化（`H(stoch)`）を有効にしたCurious Replayの比較結果である。

**スクリプト:**
- [`exp_script/251230_run_mspackman_curious.sh`](exp_script/251230_run_mspackman_curious.sh)
- [`exp_script/251231_run_mspackman_curious.sh`](exp_script/251231_run_mspackman_curious.sh)

#### 仮説
<!-- Describe the hypothesis here / ここに仮説を記述してください -->

#### 結果

![Curious Replay Entropy](./assets/mspackman_curious_ent.png)

#### 考察
<!-- Describe the discussion here / ここに考察を記述してください -->

### 3.3 TrendMixture (TrendMix)

報酬のトレンドに基づいて探索と活用をバランスさせる手法である。

**スクリプト:**
- [`exp_script/260104_run_mspackman_trendmix_multi.sh`](exp_script/260104_run_mspackman_trendmix_multi.sh)
- [`exp_script/260105_run_mspackman_trendmix_multi.sh`](exp_script/260105_run_mspackman_trendmix_multi.sh)
- [`exp_script/260106_run_mspackman_trendmix_multi.sh`](exp_script/260106_run_mspackman_trendmix_multi.sh)

#### 仮説
<!-- Describe the hypothesis here / ここに仮説を記述してください -->

#### 結果

![TrendMixture](./assets/mspackman_trendmix.png)

#### 考察
<!-- Describe the discussion here / ここに考察を記述してください -->
