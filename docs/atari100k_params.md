以下は **DreamerV3 論文（dreamerv3.pdf）**に書かれている **Atari100k（Atari 100K）** の実験設定を、該当箇所を漏れなく拾って整理したものです。 :contentReference[oaicite:0]{index=0}

---

## 1) ベンチマーク（Atari100k）そのものの定義・前提
- **タスク数**: 26本のAtariゲーム。 :contentReference[oaicite:1]{index=1}  
- **データ予算**: **400K env steps（frames）**。論文中では「**action repeat後は100K**」とも明記。 :contentReference[oaicite:2]{index=2}  
- **ゲーム時間感**: 400K framesは「**約2時間のプレイ時間**」に相当すると説明。 :contentReference[oaicite:3]{index=3}  
- **環境設定の選択**: 先行研究で環境設定の流儀が複数あることを認めつつ（詳細はTable 10に要約）、本論文では **“as originally introduced（元のAtari100kの設定）”** を採用。 :contentReference[oaicite:4]{index=4}  

---

## 2) DreamerV3側の学習設定（Atari100kでの運用）
### 計算・並列化（Table 2）
- **Action repeat**: 4 :contentReference[oaicite:5]{index=5}  
- **環境インスタンス数（並列環境）**: 1 :contentReference[oaicite:6]{index=6}  
  - 理由: Atari100kは **予算400K env steps**に対し、Atariの最大エピソード長が理論上 **432K env steps**になり得るため、単一環境を採用したと説明。 :contentReference[oaicite:7]{index=7}  
- **Replay ratio**: 128 :contentReference[oaicite:8]{index=8}  
- **モデルサイズ**: 200M parameters :contentReference[oaicite:9]{index=9}  
- **計算コスト（換算）**: 0.1 A100 GPU-days :contentReference[oaicite:10]{index=10}  
- なお本論文のDreamer/PPOは **各実験を単一のNVIDIA A100 GPUで学習**したと明記。 :contentReference[oaicite:11]{index=11}  

### シードと誤差表示
- Atari100kも含むベンチマーク全般について、DreamerとPPOは **各ベンチマーク5 seeds**で実行し、曲線は **平均＋1標準偏差**を表示。 :contentReference[oaicite:12]{index=12}  

### リプレイ（uniform replay）と replay ratio の定義（Implementation）
- Dreamerは **uniform replay buffer（オンラインキュー付き）**で実装し、ミニバッチは「オンライン軌跡→残りをリプレイから一様サンプル」で構成。 :contentReference[oaicite:13]{index=13}  
- リプレイには **latent stateも保存**して、リプレイ時の初期化に使い、学習ロールアウトで得た新しいlatentをバッファへ書き戻す。 :contentReference[oaicite:14]{index=14}  
- prioritized replayは有効だが、**本論文の実験では実装簡便性のため使わず** uniform replayを採用。 :contentReference[oaicite:15]{index=15}  
- replay ratioは「action repeatを除いた、環境1ステップ収集あたりに学習するtime step割合」で定義し、**(replay ratio) / (minibatch内time steps) / (action repeat)** で「env stepsあたりの勾配更新頻度」に換算できると説明。 :contentReference[oaicite:16]{index=16}  
  - （論文の定義に従うと）Atari100kの **replay ratio=128**, **batch shape=16×64**, **action repeat=4**なので、**勾配更新は env 32 stepごとに1回**（=128/(16·64·4)=1/32）になります（これは論文の換算式からの計算）。

### Dreamerの共通ハイパラ（Atari100kにも適用）
- **Table 4のハイパラを、全ベンチマークで同一**に使う（離散/連続、視覚/固有感覚なども含む）と明記。 :contentReference[oaicite:17]{index=17}  
  - 例: Replay capacity=5×10^6、Batch size=16、Batch length=64、Learning rate=4×10^-5、Optimizer=LaProp(ε=10^-20)、Imagination horizon=15、Discount γ に対応する horizon=333、λ=0.95 など。 :contentReference[oaicite:18]{index=18}  
- さらに **ハイパラのアニーリング、prioritized replay、weight decay、dropoutは使わない**と明記。 :contentReference[oaicite:19]{index=19}  

---

## 3) 評価の仕方・報告指標（Atari100k）
### 評価ステップ
- 結果は **400K environment steps**で報告し、これは **action repeat=4により100K agent stepsに相当**するとTable 9キャプションで明記。 :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}  

### スコアの提示形式
- 各ゲームについて **Random / Human** の参照スコアを併記し、各手法（PPO, SimPLe, SPR, TWM, IRIS, Dreamer）のスコアを **400K steps時点**で並べる（Table 9）。 :contentReference[oaicite:22]{index=22}  
- 集約指標として **Gamer mean (%)** と **Gamer median (%)** を報告（Random=0, Human=100の体裁）。 :contentReference[oaicite:23]{index=23}  
- 学習曲線（Figure 13）も提示している。 :contentReference[oaicite:24]{index=24}  

---

## 4) “Atari100kは設定が揺れやすい”ことへの注意書き（Table 10）
論文は、Atari100kは先行研究ごとに評価プロトコルが違う点を明示し、比較上の差分を **Table 10** にまとめています。 :contentReference[oaicite:25]{index=25}  
- 計算資源は **A100 GPU-days換算**で統一して列挙。 :contentReference[oaicite:26]{index=26}  
- EfficientMuZeroは最高スコアだが **標準（original）から環境設定を変更**しているため単純比較が難しい、という注記。 :contentReference[oaicite:27]{index=27}  
- IRISは **Freewayだけ探索強度ハイパラを別にしている**注記。 :contentReference[oaicite:28]{index=28}  

またTable 10では、手法ごとの「使っている要素」をX/—で示しており、Dreamerは以下を**使っていない（—）**として整理されています：
- Online planning（木探索など）なし :contentReference[oaicite:29]{index=29}  
- Data augmentation なし :contentReference[oaicite:30]{index=30}  
- Non-uniform replay（優先度付きなど）なし :contentReference[oaicite:31]{index=31}  
- Separate hparams（タスク別など）なし :contentReference[oaicite:32]{index=32}  
- Increased resolution なし :contentReference[oaicite:33]{index=33}  
- Uses life information なし :contentReference[oaicite:34]{index=34}  
- Uses early resets なし :contentReference[oaicite:35]{index=35}  
- Separate eval episodes なし :contentReference[oaicite:36]{index=36}  

---

## 5) 論文が言及している比較対象（Atari100kの文脈）
- 本文中では、SOTAとしてEfficientZero（オンライン探索＋prioritized replay＋ハイパラschedule、さらにearly reset）に触れつつ、複雑さなしの比較対象として **IRIS / TWM / SPR / SimPLe** 等に対する優位性を述べています。 :contentReference[oaicite:37]{index=37}  
- 実際にTable 9で並べている比較列もこれに対応（PPO, SimPLe, SPR, TWM, IRIS, Dreamer）。 :contentReference[oaicite:38]{index=38}  

---

必要なら、この設定を「(A) 先行研究のAtari100k設定の揺れ（Table 10の含意）」「(B) DreamerV3が“揺れなし設定”で勝っていると言える範囲」みたいに、比較の公正性の観点で整理し直すこともできます。
