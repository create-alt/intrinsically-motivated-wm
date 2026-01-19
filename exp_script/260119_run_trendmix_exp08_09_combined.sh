#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/trendmix_exp08_09_combined_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "TrendMixture exp08 + exp09: 10 experiments started in background"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
    echo ""
    echo "Experiments (sequential):"
    echo "  [exp08] curious replay priority (5 games)"
    echo "  [exp09] curious + velocity-accel hybrid (5 games)"
    echo ""
    echo "Useful commands:"
    echo "  Monitor:       tail -f $MASTER_LOG"
    echo "  Check process: ps aux | grep $PID"
    echo "  Stop all:      kill $PID"
    echo "=========================================="
    exit 0
fi

# 以下はバックグラウンドで実行される部分
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

echo "=========================================="
echo "Starting TrendMixture exp08 + exp09 combined experiments"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

###############################################################################
# exp08: curious replay priority
###############################################################################
echo "=========================================="
echo "Running exp08 (curious replay priority)..."
echo "=========================================="
bash exp_script/260119_run_trendmix_exp08_5games_curious.sh --running

echo ""
echo "=========================================="
echo "exp08 completed at: $(date)"
echo "=========================================="
echo ""

###############################################################################
# exp09: curious + velocity-accel hybrid
###############################################################################
echo "=========================================="
echo "Running exp09 (curious + hybrid)..."
echo "=========================================="
bash exp_script/260119_run_trendmix_exp09_5games_curious_hybrid.sh --running

echo ""
echo "=========================================="
echo "exp09 completed at: $(date)"
echo "=========================================="

###############################################################################
echo ""
echo "=========================================="
echo "All experiments (exp08 + exp09) completed at: $(date)"
echo "=========================================="
