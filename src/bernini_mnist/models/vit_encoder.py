import torch
import torch.nn as nn
from typing import Tuple, Optional

class MNISTViT(nn.Module):
    """
    16-Patch Vision Transformer for MNIST classification and continuous semantic feature extraction.
    
    Given a 28x28 image, it splits into 4x4 = 16 non-overlapping 7x7 patches.
    It outputs a sequence of N=16 continuous semantic embeddings of dimension `embed_dim` (default 256).
    """
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # (28//7)^2 = 16
        self.embed_dim = embed_dim

        # Patch embedding via 2D convolution
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable 1D positional embeddings for the 16 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head on pooled token representation
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the N=16 continuous semantic patch tokens from input images.
        
        Args:
            x: Tensor of shape (B, 1, 28, 28)
        Returns:
            features: Tensor of shape (B, 16, embed_dim)
        """
        # Patch projection: (B, 1, 28, 28) -> (B, embed_dim, 4, 4)
        x = self.patch_embed(x)
        # Flatten spatial dimensions: (B, embed_dim, 16) -> (B, 16, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)
        # Transformer forward pass
        features = self.transformer(x)
        features = self.norm(features)
        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits and intermediate 16 patch features.
        
        Args:
            x: Tensor of shape (B, 1, 28, 28)
        Returns:
            logits: Tensor of shape (B, 10)
            features: Tensor of shape (B, 16, embed_dim)
        """
        features = self.extract_features(x)
        # Global average pooling across the 16 tokens for classification
        pooled = features.mean(dim=1)
        logits = self.head(pooled)
        return logits, features
