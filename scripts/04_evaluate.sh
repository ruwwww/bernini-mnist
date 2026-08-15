#!/usr/bin/env bash
set -e
export PYTHONPATH=src

echo "=== Stage 5: Quantitative Evaluation of Bernini MNIST Pipeline ==="
python3 scripts/04_evaluate.py \
    --data_dir ./data \
    --mllm_model_name Qwen/Qwen3-0.6B \
    --vit_path ./checkpoints/vit_mnist.pt \
    --planner_path ./checkpoints/planner_qwen.pt \
    --renderer_path ./checkpoints/renderer_semantic.pt \
    --output_dir ./outputs/evaluation \
    --samples_per_class 30 \
    --refinement_steps 4
