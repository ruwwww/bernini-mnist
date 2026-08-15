#!/usr/bin/env bash
set -e
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage 3: Joint Fine-Tuning of Planner & 2D Renderer ==="
python3 -m bernini_mnist.training.train_joint \
    --data_dir ./data \
    --mllm_model_name Qwen/Qwen3-0.6B \
    --vit_path ./checkpoints/vit_mnist.pt \
    --planner_path ./checkpoints/planner_qwen.pt \
    --renderer_path ./checkpoints/renderer_semantic.pt \
    --save_path ./checkpoints/joint_pipeline.pt \
    --sample_dir ./outputs/samples_joint \
    --batch_size 64 \
    --epochs 8 \
    --lr 1e-4 \
    --lambda_plan 1.0 \
    --lambda_render 2.0 \
    --renderer_hidden 128 \
    --sample_every 4
