import math
import torch
import torch.nn as nn
from typing import Optional, Union

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps in [0, 1] into vector representations."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-torch.arange(0, half, dtype=torch.float32, device=t.device) * (9.210340371976184 / half))
        args = t[:, None].float() * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(t_freq)

class ConvResBlock2D(nn.Module):
    """2D Residual Convolutional Block with GroupNorm and SiLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class MLPFlowRenderer(nn.Module):
    """
    2D Spatial Flow Matching Denoiser / Renderer for MNIST (28x28).
    
    Preserves 2D spatial locality by mapping the 16 continuous patch tokens
    (corresponding to a 4x4 spatial grid of 7x7 patches) directly to pixel space.
    """
    def __init__(
        self,
        in_dim: int = 784,
        hidden_size: int = 128,
        num_layers: int = 4,
        semantic_dim: int = 256,
        num_semantic_tokens: int = 16,
        num_classes: int = 10,
        cond_type: str = "semantic"  # "semantic" or "class"
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.cond_type = cond_type
        self.semantic_dim = semantic_dim

        self.time_embed = TimestepEmbedder(hidden_size)
        self.in_conv = nn.Conv2d(1, hidden_size, 3, padding=1)

        if cond_type == "semantic":
            # Smooth spatial token upsampling: (B, 256, 4, 4) -> (B, hidden_size, 28, 28)
            self.cond_up = nn.Sequential(
                nn.Conv2d(semantic_dim, hidden_size, 3, padding=1),
                nn.SiLU(),
                nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False),
                nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
            )
        elif cond_type == "class":
            self.class_embed = nn.Embedding(num_classes, hidden_size)
        else:
            raise ValueError(f"Unknown cond_type: {cond_type}")

        self.blocks = nn.ModuleList([
            ConvResBlock2D(hidden_size) for _ in range(num_layers)
        ])

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, 1, 3, padding=1)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict pixel velocity vector field v_phi(x_t, t, cond).
        
        Args:
            x_t: Tensor of shape (B, 784) or (B, 1, 28, 28)
            t: Timestep tensor of shape (B,)
            cond: Semantic tokens (B, 16, 256) or class labels (B,)
            
        Returns:
            v_pred: Predicted velocity of shape (B, 784)
        """
        if x_t.dim() == 2:
            x_t = x_t.view(-1, 1, 28, 28)

        B = x_t.shape[0]
        t_feat = self.time_embed(t).view(B, -1, 1, 1)

        if self.cond_type == "semantic":
            # (B, 16, 256) -> transpose to (B, 256, 4, 4)
            cond_map = cond.transpose(1, 2).view(B, self.semantic_dim, 4, 4)
            c_feat = self.cond_up(cond_map)
        else:
            c_emb = self.class_embed(cond).view(B, -1, 1, 1)
            c_feat = c_emb.expand(-1, -1, 28, 28)

        x = self.in_conv(x_t) + c_feat + t_feat
        for block in self.blocks:
            x = block(x)

        v_pred_2d = self.out_conv(x)
        return v_pred_2d.flatten(1)
