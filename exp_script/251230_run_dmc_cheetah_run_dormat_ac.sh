#!/bin/bash

cd /home/ist_baidoku/yoshinari.kawashima/wm25_final_homework/dreamerv3

# 環境変数を読み込む
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

# 仮想環境をアクティベート
source .venv/bin/activate

# タイムスタンプ
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_ac_dmc_cheetah_run_vision_12m"

# ログディレクトリ作成
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Starting training in background"
echo "Log directory: ${LOG_DIR}"
echo "Log file: ${LOG_DIR}/log.log"
echo "=========================================="
echo ""

# GPU情報の確認と保存
echo "GPU Status:"
nvidia-smi
echo ""

# GPU情報をログファイルにも保存
nvidia-smi > ${LOG_DIR}/log.log 2>&1
echo "" >> ${LOG_DIR}/log.log
echo "=========================================" >> ${LOG_DIR}/log.log
echo "Training started at $(date)" >> ${LOG_DIR}/log.log
echo "=========================================" >> ${LOG_DIR}/log.log
echo "" >> ${LOG_DIR}/log.log

# nohupで実行
nohup python dreamerv3/main.py \
    --configs dmc_vision size12m \
    --task dmc_cheetah_run \
    --run.steps 1e6 \
    --run.envs 16 \
    --run.train_ratio 512 \
    --env.dmc.repeat 2 \
    --logdir ${LOG_DIR} \
    --seed 0 \
    --method ac \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    > ${LOG_DIR}/log.log 2>&1 &

PID=$!

echo ""
echo "✓ Training started successfully"
echo "  PID: ${PID}"
echo "  Log file: ${LOG_DIR}/log.log"
echo ""
echo "Useful commands:"
echo "  Monitor log:     tail -f ${LOG_DIR}/log.log"
echo "  Check process:   ps aux | grep ${PID}"
echo "  Stop training:   kill ${PID}"
echo "=========================================="
