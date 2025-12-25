
# 論文のハイパラ

## タスク

| Benchmark Tasks | Env | Steps | Action Repeat | Env Instances | Replay Ratio | GPU Days | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Minecraft | 1 | 100M | 1 | 64 | 32 | 8.9 | 200M |
| DMLab | 30 | 100M | 4 | 16 | 32 | 2.9 | 200M |
| ProcGen | 16 | 50M | 1 | 16 | 64 | 16.1 | 200M |
| Atari | 57 | 200M | 4 | 16 | 32 | 7.7 | 200M |
| Atari100K | 26 | 400K | 4 | 1 | 128 | 0.1 | 200M |
| BSuite | 23 | — | 1 | 1 | 1024 | 0.5 | 200M |
| Proprio Control | 18 | 500K | 2 | 16 | 512 | 0.3 | 12M |
| Visual Control | 20 | 1M | 2 | 16 | 512 | 0.1 | 12M |

## モデルサイズ

| Parameters | 12M | 25M | 50M | 100M | 200M | 400M |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Hidden size (d) | 256 | 384 | 512 | 768 | 1024 | 1536 |
| Recurrent units (8d) | 1024 | 3072 | 4096 | 6144 | 8192 | 12288 |
| Base CNN channels (d/16) | 16 | 24 | 32 | 48 | 64 | 96 |
| Codes per latent (d/16) | 16 | 24 | 32 | 48 | 64 | 96 |

## その他

| Name | Symbol | Value |
| :--- | :--- | :--- |
| **General** | | |
| Replay capacity | — | 5 × 10^6 |
| Batch size | B | 16 |
| Batch length | T | 64 |
| Activation | — | RMSNorm + SiLU |
| Learning rate | — | 4 × 10^-5 |
| Gradient clipping | — | AGC(0.3) |
| Optimizer | — | LaProp(ϵ = 10^-20) |
| **World Model** | | |
| Reconstruction loss scale | βpred | 1 |
| Dynamics loss scale | βdyn | 1 |
| Representation loss scale | βrep | 0.1 |
| Latent unimix | — | 1% |
| Free nats | — | 1 |
| **Actor Critic** | | |
| Imagination horizon | H | 15 |
| Discount horizon | 1/(1 − γ) | 333 |
| Return lambda | λ | 0.95 |
| Critic loss scale | βval | 1 |
| Critic replay loss scale | βrepval | 0.3 |
| Critic EMA regularizer | — | 1 |
| Critic EMA decay | — | 0.98 |
| Actor loss scale | βpol | 1 |
| Actor entropy regularizer | η | 3 × 10^-4 |
| Actor unimix | — | 1% |
| Actor RetNorm scale | S | Per(R, 95) − Per(R, 5) |
| Actor RetNorm limit | L | 1 |
| Actor RetNorm decay | — | 0.99 |