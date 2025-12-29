# Curious Replay（Kauvar et al., 2023）仕様（論文準拠）

## 0. 目的と適用箇所
- Curious Replay（CR）は **行動選択（exploration）ではなく**、Dreamer系の **world model 学習時にリプレイバッファから何をサンプルするか**を変える手法。
- ねらいは「環境が変化した後に、world model が新しいデータを十分に学べず追従が遅い」問題に対して、
  - (a) **まだ学習にあまり使っていない経験** と
  - (b) **world model がまだうまく予測できていない経験**
  を優先して world model を更新すること。

---

## 1. 用語とデータ単位
### 1.1 経験（experience）
- 論文の基本定義では、experience は **単一の状態遷移（state transition）**（例： (o_t, a_t, r_t, o_{t+1}, done) ）を指す。

### 1.2 追跡するメタ情報（各 experience i に付与）
- **再生回数（visit / replay count）**:  v_i  
  - 「experience i が **学習バッチに含まれて world model 更新に使われた回数**」。
- **優先度（priority）**: p_i  
  - サンプリング確率を決めるスカラー。

### 1.3 world model 損失（Dreamerの場合）
- Adversarial成分で使う損失 L は **world model を学習するときの損失**。
- Dreamerでは論文中で次を使用：
  - L = L_image + L_reward + L_KL
- CRはこの L を experience 単位に割り当てる（バッチ学習時に、各 experience の損失をキャッシュして優先度更新に使う）。

---

## 2. ハイパーパラメータ
- β ∈ [0, 1] : count-based の減衰率（v_i が増えるほど β^{v_i} が小さくなる）
- α ∈ [0, 1] : adversarial（損失ベース）優先度の鋭さ（α=0 なら一様）
- ϵ > 0 : (|L|+ϵ) の安定化定数
- c ≥ 0 : count-based 項のスケール係数（count-based と adversarial の相対重み）
- E : environment steps per train step（環境相互作用 E ステップごとに学習を1回回す、という運用パラメータ）
- B : 学習でサンプルするバッチサイズ
- p_max : 新規 experience に付与する「最大優先度」（初期化用）

---

## 3. リプレイバッファとサンプリング分布
### 3.1 収納
- バッファ容量を |R| とする（有限）。
- 各 experience i を、(データ本体, v_i, p_i) として保持する。

### 3.2 サンプリング確率
- バッファからのサンプルは **優先度 p_i に比例**させる：
  - P(i) = p_i / (Σ_j p_j)
- 実装上は SumTree 等で正規化サンプリングを効率化する（論文では SumTree を用いる前提でアルゴリズムを書いている）。

---

## 4. 優先度の定義
### 4.1 Count-based Replay（単体）
- count-based の優先度は
  - p_i = β^{v_i}
- 意味：学習に使われた回数 v_i が少ない（新しい／まだ十分学習していない）経験ほど優先される。

### 4.2 Adversarial Replay（単体）
- adversarial（損失ベース）の優先度は
  - p_i = (|L_i| + ϵ)^{α}
- 重要な更新規則：
  - **優先度は “その経験を学習に使ったときだけ” 更新する**（全バッファを毎回再計算しない）。
  - 新規 experience の優先度は **最大値で初期化**する（p_i ← p_max）。

### 4.3 Curious Replay（Count + Adversarial の加算）
- CR の最終優先度は加算で統合：
  - p_i = c * β^{v_i} + (|L_i| + ϵ)^{α}
- 直感：
  - c * β^{v_i} が「未学習・新規データを押し上げる」
  - (|L_i| + ϵ)^{α} が「難しい（モデルが外している）データを押し上げる」

---

## 5. 学習ループ仕様（Algorithm 1 相当）
各 iteration で以下を繰り返す。

### 5.1 環境相互作用（collect）
- 現在のポリシーで環境から遷移を収集する（運用としては E ステップごとに学習へ進む）。

### 5.2 バッファ追加（append）
- 収集した各新規遷移 i について
  - v_i ← 0
  - p_i ← p_max（「最大優先度」で初期化）
  - バッファへ追加（容量超過時は通常の方式で古いものから削除等。※論文は削除ポリシー自体は規定しない）

### 5.3 サンプリング（sample）
- 優先度に比例した確率 P(i) で、バッファから B 個の experience をサンプルする。

### 5.4 学習（train）
- サンプルしたバッチで Dreamer の通常手順どおり
  - world model を更新し
  - （Dreamerの枠組みとして）actor-critic を更新する
- このとき **experience ごとの損失 L_i を計算してキャッシュ**する
  - ここでの L_i は world model 学習損失（Dreamerなら L_image + L_reward + L_KL）に基づく。

### 5.5 優先度・カウント更新（update）
- **バッチに含まれた各 experience i** についてのみ、次を行う：
  1) v_i ← v_i + 1  
  2) p_i ← c * β^{v_i} + (|L_i| + ϵ)^{α}
- 注意（論文の明示点）：
  - 「学習に使われたときだけ優先度更新」なので、未サンプルの experience の優先度は古いまま（stale priority の可能性）。
  - ただし論文では、その stale が問題になる兆候は実験で観測していない。

---

## 6. DreamerV3 実装に関する追加仕様（論文の “Implementation details”）
DreamerV3はリプレイが「遷移」ではなく「シーケンス（trajectory chunk）」単位なので、CRを次のように適用する。

### 6.1 バッファの格納単位
- replay buffer には **長さ 64 のシーケンス**を格納する。

### 6.2 シーケンスのサンプリング確率の決め方
- **シーケンスを学習に使う確率は、そのシーケンスの “最後のステップ” の優先度**で決める。
  - （言い換え）sequence-level priority = priority(last step)

### 6.3 学習後の更新対象
- 1回の training step の後、
  - **そのシーケンス内の各ステップ**について、v と p を更新する
  - （=「最後のステップだけ」ではなく、サンプルされたシーケンスに含まれる全ステップを “trained on” として扱う）

### 6.4 実装コンポーネント
- DreamerV3 側では **Reverb replay buffer**を用いて、シーケンスと優先度を保持しサンプリングする。

### 6.5 running minimum の扱い
- DreamerV3 の CR 実装では **loss を running minimum で補正しない**（DreamerV2実装と異なる）。

---

## 7. DreamerV2 / DreamerPro 実装に関する追加仕様（論文の “Implementation details”）
- DreamerV2 / DreamerPro では SumTree 実装として STArr を利用。
- Curious Replay の優先度計算に使う loss について：
  - **（全学習を通した）running minimum を loss から引いた値**を使ってから priority を計算する運用をしている。
  - （注意）Temporal-Difference 優先度（比較用）では running minimum を使わない。

---

## 8. “論文と差が出やすい”実装上の注意点（仕様として明示）
- 優先度更新は **「サンプルして学習に使った experience（または sampled sequence 内の各 step）」だけ**。全バッファを毎回更新しない。
- 新規 experience の優先度は **常に最大値 p_max**で初期化する（小さく初期化しない）。
- DreamerV3 では **sequence sampling = last step priority**、ただし **update は sequence 内の全 step**。
- DreamerV2/Pro では **running minimum 補正あり**、DreamerV3 では **なし**（論文記載の差分）。

---
