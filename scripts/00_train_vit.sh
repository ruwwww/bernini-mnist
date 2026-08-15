#!/usr/bin/env bash
set -e
export PYTHONPATH=src

echo "=== Stage 0: Training MNIST 16-Patch ViT Oracle ==="
python3 -m bernini_mnist.training.train_vit \
    --data_dir ./data \
    --save_path ./checkpoints/vit_mnist.pt \
    --batch_size 128 \
    --epochs 10 \
    --lr 1e-3 \
    --embed_dim 256 \
    --depth 4 \
    --num_heads 4
