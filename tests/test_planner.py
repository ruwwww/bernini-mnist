import torch
import pytest
from bernini_mnist.models.planner import SemanticFlowMatchingHead, BerniniSemanticPlanner

def test_semantic_flow_matching_head():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = SemanticFlowMatchingHead(
        semantic_dim=256,
        qwen_hidden_size=1024,
        head_hidden_size=256,
        num_layers=2
    ).to(device)
    
    z_t = torch.randn(4, 16, 256, device=device)
    t = torch.rand(4, device=device)
    h_qwen = torch.randn(4, 16, 1024, device=device)
    
    v_pred = head(z_t, t, h_qwen)
    assert v_pred.shape == (4, 16, 256), f"Expected shape (4, 16, 256), got {v_pred.shape}"

def test_planner_forward_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = BerniniSemanticPlanner(
        mllm_model_name="Qwen/Qwen3-0.6B",
        semantic_dim=256,
        num_semantic_tokens=16,
        num_classes=10,
        head_hidden_size=256,
        head_layers=2,
        freeze_mllm=True
    ).to(device)
    
    class_labels = torch.tensor([0, 3, 5, 9], device=device)
    z_1 = torch.randn(4, 16, 256, device=device)
    
    loss = planner.compute_loss(class_labels, z_1, mask_ratio=0.5)
    assert loss.dim() == 0, f"Expected scalar loss, got {loss.shape}"
    assert not torch.isnan(loss), "Loss was NaN"
