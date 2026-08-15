import torch
import pytest
from bernini_mnist.models.renderer import MLPFlowRenderer

def test_semantic_flow_renderer_forward():
    renderer = MLPFlowRenderer(
        in_dim=784,
        hidden_size=256,
        num_layers=3,
        semantic_dim=256,
        num_semantic_tokens=16,
        cond_type="semantic"
    )
    
    x_t = torch.randn(4, 784)
    t = torch.rand(4)
    cond = torch.randn(4, 16, 256)
    
    v_pred = renderer(x_t, t, cond)
    assert v_pred.shape == (4, 784), f"Expected shape (4, 784), got {v_pred.shape}"

def test_class_flow_renderer_forward():
    renderer = MLPFlowRenderer(
        in_dim=784,
        hidden_size=256,
        num_layers=3,
        num_classes=10,
        cond_type="class"
    )
    
    x_t = torch.randn(4, 1, 28, 28)  # Test with 4D image input
    t = torch.rand(4)
    labels = torch.tensor([0, 3, 7, 9])
    
    v_pred = renderer(x_t, t, labels)
    assert v_pred.shape == (4, 784), f"Expected shape (4, 784), got {v_pred.shape}"
