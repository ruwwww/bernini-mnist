import torch
import pytest
from bernini_mnist.models.pipeline import BerniniMNISTPipeline
from bernini_mnist.models.planner import BerniniSemanticPlanner
from bernini_mnist.models.renderer import MLPFlowRenderer

def test_pipeline_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = BerniniSemanticPlanner(
        mllm_model_name="Qwen/Qwen3-0.6B",
        semantic_dim=256,
        num_semantic_tokens=16,
        head_hidden_size=128,
        head_layers=2,
        freeze_mllm=True
    ).to(device)
    
    renderer = MLPFlowRenderer(
        in_dim=784,
        hidden_size=128,
        num_layers=2,
        semantic_dim=256,
        num_semantic_tokens=16,
        cond_type="semantic"
    ).to(device)
    
    pipeline = BerniniMNISTPipeline(planner=planner, renderer=renderer).to(device)
    
    labels = torch.tensor([1, 7], device=device)
    images, semantic_tokens = pipeline.generate(
        labels,
        num_refinement_steps=2,
        ode_steps_planner=5,
        ode_steps_renderer=5
    )
    
    assert images.shape == (2, 1, 28, 28)
    assert semantic_tokens.shape == (2, 16, 256)
    assert (images >= 0.0).all() and (images <= 1.0).all()
