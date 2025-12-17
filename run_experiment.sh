#!/bin/bash

# プロジェクトルートに移動
cd /home/ist_baidoku/yoshinari.kawashima/wm25_final_homework/dreamerv3

# 環境変数の読み込み (.envファイルが存在する場合)
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found. WandB may not be configured."
fi

# 仮想環境をアクティベート
source .venv/bin/activate

# 実験設定
EXP_NAME="dreamerV3_atari100"
TASK="atari100k_seaquest"
CONFIG="atari100k"
SEED=0

# タイムスタンプ付きログディレクトリ
TIME_STR=$(date '+%Y_%m_%d_%H%M')
LOG_DIR="log/${EXP_NAME}_${TIME_STR}"

# ログディレクトリ作成
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "実験名: ${EXP_NAME}"
echo "タスク: ${TASK}"
echo "コンフィグ: ${CONFIG}"
echo "シード: ${SEED}"
echo "ログディレクトリ: ${LOG_DIR}"
echo "=========================================="

# WandB設定の確認
if [ -n "$WANDB_API_KEY" ]; then
    echo "✓ WandB API Key is set"
    echo "  Project: ${WANDB_PROJECT:-default}"
    echo "  Entity: ${WANDB_ENTITY:-default}"
else
    echo "⚠ WandB API Key not found. Set it in .env file."
fi

echo "=========================================="
echo ""

# GPU情報の確認
echo "GPU Status:"
echo "=========================================="
nvidia-smi
echo "=========================================="
echo ""

# 訓練実行
python dreamerv3/main.py \
    --configs ${CONFIG} \
    --task ${TASK} \
    --logdir ${LOG_DIR} \
    --seed ${SEED} \
    --jax.platform cuda \
    --logger.outputs jsonl,tensorboard,wandb

echo ""
echo "=========================================="
echo "訓練完了: ${LOG_DIR}"
echo "=========================================="
