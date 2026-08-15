#!/usr/bin/env bash
set -e
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage 1: Training Bernini Semantic Planner (DiT + Qwen3-0.6B) ==="
python3 -m bernini_mnist.training.train_planner \
    --data_dir ./data \
    --mllm_model_name Qwen/Qwen3-0.6B \
    --vit_path ./checkpoints/vit_mnist.pt \
    --renderer_path ./checkpoints/renderer_semantic.pt \
    --save_path ./checkpoints/planner_qwen.pt \
    --sample_dir ./outputs/samples_planner \
    --batch_size 128 \
    --epochs 12 \
    --lr 3e-4 \
    --mask_ratio 0.5 \
    --head_hidden_size 512 \
    --renderer_hidden 128 \
    --sample_every 4
