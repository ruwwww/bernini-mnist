#!/usr/bin/env bash
set -e
export PYTHONPATH=src

echo "=== Stage 2: Training 2D Spatial Flow Matching Renderer ==="
python3 -m bernini_mnist.training.train_renderer \
    --data_dir ./data \
    --vit_path ./checkpoints/vit_mnist.pt \
    --save_path ./checkpoints/renderer_semantic.pt \
    --sample_dir ./outputs/samples_renderer \
    --cond_type semantic \
    --batch_size 256 \
    --epochs 12 \
    --lr 1e-3 \
    --hidden_size 128 \
    --num_layers 4 \
    --sample_every 4
