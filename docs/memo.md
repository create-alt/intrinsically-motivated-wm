# DreamerV3 実験メモ

## 環境構築

```bash
# 仮想環境作成とインストール
uv venv
uv pip install -e .
uv pip install -e ".[atari]"  # Atari環境が必要な場合
```

## WandB設定

```bash
# 1. .envファイルを作成
cp .env.example .env

# 2. .envを編集（APIキーとプロジェクト名を設定）
nano .env

# 必須項目:
# WANDB_API_KEY=your_api_key  # https://wandb.ai/authorize
# WANDB_PROJECT=dreamerv3_jax
```

## 実験実行

### 対話的実行
```bash
./run_experiment.sh
```

### バックグラウンド実行（SSH切断してもOK）
```bash
./run_nohup.sh

# ログ確認
tail -f log/dreamerV3_atari100_*.log
```

### 手動実行
```bash
# 環境変数を読み込む
export $(cat .env | grep -v '^#' | xargs)
source .venv/bin/activate

# 実行
python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_pong \
    --logdir log/exp_$(date '+%Y_%m_%d_%H%M') \
    --seed 0 \
    --jax.platform cuda \
    --logger.outputs jsonl,tensorboard,wandb
```

## よく使うコマンド

```bash
# プロセス確認
ps aux | grep dreamerv3

# プロセス停止
pkill -f dreamerv3

# ログ確認
tail -f log/*.log
tail -f log/*/metrics.jsonl

# GPU確認
nvidia-smi
watch -n 1 nvidia-smi  # 1秒ごとに更新
```

## 設定変更

### タスク変更
`run_experiment.sh` または `run_nohup.sh` の以下を編集：
```bash
TASK="atari100k_breakout"  # pong → breakout
CONFIG="atari100k"
SEED=0
```

### その他の設定
- 全設定: `dreamerv3/configs.yaml`
- コマンドラインで上書き可能: `--batch_size 16`



実行した操作：
1. 現在のmainブランチを backup-main ブランチに保存
2. mainブランチをコミット 83271ee (Merge branch 'start_exp' into main) に移動

現在の状態：
- mainブランチ：コミット 83271ee を指しています
- backup-mainブランチ：元のmain（コミット 08b9d29）を保持しています

今後の作業：
このコミットから通常通り作業を開始できます。

元に戻したい場合：
# バックアップに戻す
git reset --hard backup-main

# または、バックアップブランチに切り替える
git checkout backup-main

バックアップが不要になったら：
# バックアップブランチを削除
git branch -D backup-main

これでコミット 83271ee から安全に作業を開始できます！