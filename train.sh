#!/usr/bin/env bash

set -euo pipefail

python3 main.py \
  train \
  --epochs 10000 \
  --lr 0.001 \
  --selfplay_games_per_epoch 1 \
  --mcts_sims 100 \
  --updates_per_epoch 10 \
  --batch 64 \
  --channels 32 \
  --blocks 3 \
  --board_size 5 \
  --komi 0.5 \
  --buffer 10000 \
  --augment \
  --temp_moves 3 \
  --ckpt checkpoints/5x5_az.pt \
  --cpu
