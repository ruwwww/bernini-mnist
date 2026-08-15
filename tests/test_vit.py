import torch
import pytest
from bernini_mnist.models.vit_encoder import MNISTViT

def test_vit_encoder_shapes():
    model = MNISTViT(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=256,
        depth=4,
        num_heads=4
    )
    
    x = torch.randn(4, 1, 28, 28)
    logits, features = model(x)
    
    assert logits.shape == (4, 10), f"Expected logits shape (4, 10), got {logits.shape}"
    assert features.shape == (4, 16, 256), f"Expected features shape (4, 16, 256), got {features.shape}"
    
    extracted = model.extract_features(x)
    assert extracted.shape == (4, 16, 256), f"Expected extracted shape (4, 16, 256), got {extracted.shape}"
