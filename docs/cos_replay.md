# DreamerV3：リプレイバッファのサンプリングを「類似度優先」にする実装まとめ（コード無し）

## ゴール
DreamerV3の学習手順（world model / actor / critic、online queue 混合など）は維持しつつ、**Replay Buffer からのサンプリング確率だけ**を変更する。

- 画像↔画像の類似度：**DINOv2**
- 画像↔言語の類似度：**CLIP / OpenCLIP**
- cos類似度からスコア → priority → サンプル確率を作り、Replayから引かれる頻度（期待値）を変える。

---

## 1. 変える場所（最小改変）
### 1.1 Replayから引く部分のみ差し替え
- DreamerV3は通常、ミニバッチを
  - **online queue（直近の新鮮な軌跡）**から一部
  - **replay buffer（過去の軌跡）**から残り
 で構成する。
- この設計は保持し、**replay buffer の抽選のみ**を
  - 既存：一様サンプル
  - 変更：優先度（priority）に比例した確率でサンプル
に差し替える。

### 1.2 サンプリング単位は trajectory chunk のまま
- DreamerV3で学習に使う単位（例：長さL=64のシーケンス）に合わせ、priorityも**chunk単位**で持つ。
- transition単位のPERにしない（実装/整合性/コスト面で不利になりやすい）。

---

## 2. Replay item（1チャンク）に追加で持つメタ情報
### 2.1 埋め込み（embedding）のキャッシュ
- `embed`: L2正規化済みの特徴ベクトル（次元D）
- 計算タイミング：**chunkが確定してreplayに入る瞬間に1回だけ**（推奨）
  - 学習ステップごとに再計算しない（速度が崩れる）
- 代表フレーム：まずは **末尾フレーム1枚**（最小コスト）
  - 代替：ランダム1枚／平均（精度と計算コストのトレードオフ）

### 2.2 priority（サンプリング用スカラー）
- `prio`: 非負スカラー（後述のscoreから算出）
- リプレイ抽選は `prio` で決まるので、これを保存しておく。

---

## 3. 類似度スコアの定義（cos類似度）
共通前提：
- すべての埋め込みは **L2正規化**しておく（cosine = 内積）
- `pos`（注目すべき / 成功 / 学びたい）と `neg`（noisy TV / 注目しない）は「参照バンク」として複数ベクトル（K個）を持てる。

### 3.1 画像↔画像（DINOv2）
- 観測画像 → DINOv2 → `e_img`（正規化）
- 参照：`pos_bank`（成功画像の埋め込み集合）、任意で `neg_bank`
- score（例）：
  - posのみ：`score = max_k cos(e_img, pos_bank[k])`
  - pos-neg差分：`score = relu(max cos(e_img,pos_bank) - max cos(e_img,neg_bank))`

### 3.2 画像↔言語（CLIP / OpenCLIP）
- 観測画像 → CLIP image encoder → `e_img`（正規化）
- テキストは開始時に一度だけエンコードして固定：
  - `t_pos = CLIP_text(prompt_pos)`（正規化）
  - `t_neg = CLIP_text(prompt_neg)`（任意、正規化）
- score（例）：
  - posのみ：`score = cos(e_img, t_pos)`
  - pos-neg差分：`score = relu(cos(e_img,t_pos) - cos(e_img,t_neg))`

---

## 4. score → priority → サンプル確率（頻度への反映）
### 4.1 priority（PER風）
- `prio = (eps + score)^alpha`
  - `eps`：ゼロ回避の小定数（例：1e-4）
  - `alpha`：優先度の強さ（例：0.3〜1.0）
- 直感：
  - alphaを大きくすると「高スコアをさらに引きやすい」
  - alphaを小さくすると「優先の効きが弱い（安定）」

### 4.2 Replayからのサンプル確率
- Replay内の全アイテム i について：
  - `P_prio(i) = prio_i / sum(prio)`
- 任意で一様分布を混ぜて安定化（推奨）：
  - `P(i) = λ*(1/N) + (1-λ)*P_prio(i)`
  - λ（例：0.1〜0.5、まず0.2）
- これにより、scoreが高いchunkほど **Replayから選ばれる確率が上がり、期待リプレイ頻度が増える**：
  - 1回の学習でReplayからM個引くなら、chunk i の期待出現回数は `M*P(i)`。

> 注意：ここでいう「softmax」は不要。実装は基本「正規化（割り算）」でOK。

---

## 5. 参照バンク（pos/neg）の作り方
### 5.1 手動（導入が簡単）
- pos_bank：成功状態の画像を少数与える（あるいは成功っぽい画像）
- neg_bank：noisy TV/無関係背景の画像（またはテキストprompt）
- 画像の場合はDINOv2で埋め込み化して保存
- テキストの場合はCLIPで埋め込み化して保存

### 5.2 自己生成（リークが少なくRLらしい）
- pos_bank：過去の経験から高return chunkの代表フレームを集める
- neg_bank：低return、distractor強、背景変化大などから集める
- 更新頻度：一定ステップごとに再構築（例：10k env steps）
- 簡易版：バンク更新しても既存アイテムのprioは再計算しない（まず動く）
  - 厳密版：prioの再計算（全体or部分）まで設計

---

## 6. 実装上の速度・安定性の要点
### 6.1 速度を落とさない原則
- 埋め込み計算（DINO/CLIP）は重いので、**chunk追加時に1回だけ**計算してキャッシュする。
- 学習時は基本「prioに基づく抽選」だけにする（cos類似度再計算はしない）。
- Replayの抽選は、最初は簡単実装（O(N)）でも動くが、大規模なら
  - SumTree / SegmentTree / Alias法などで高速化する。

### 6.2 安定化の原則
- 一様混合（λ）を入れて偏りすぎを防ぐ（モード崩壊・多様性喪失対策）。
- alphaは控えめから始めて段階的に上げる。
- pos-neg差分（noisy抑制）を使うと「背景/ノイズ」への引っ張られが減りやすい。

---

## 7. 使うライブラリ（推奨）
### 7.1 DINOv2（画像↔画像）
- 推奨：**Hugging Face Transformers**
  - 前処理（リサイズ・正規化）とモデルI/Fが揃っており、RL統合が楽。
- 代替：公式repo（torch.hub）
  - 原典に近いが前処理/出力の合わせ込みが増えがち。

### 7.2 CLIP（画像↔言語）
- 推奨：**OpenCLIP（open_clip）**
  - 多様な事前学習重みがあり、モデルサイズ比較が容易。
- 代替：Hugging Face Transformers（CLIPModel/Processor）
  - I/Fを統一したい場合に有利。

---

## 8. 推奨の最小構成（まず動かす）
- DINOv2（画像↔画像）で開始（安定＆軽い）
- 代表フレーム：chunk末尾1枚
- `alpha=0.6, eps=1e-4, λ=0.2`
- online queue混合は既存設定を維持
- 比較：DreamerV3（既存） vs DreamerV3（replayのみ優先サンプル）
