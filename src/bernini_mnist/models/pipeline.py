import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .vit_encoder import MNISTViT
from .planner import BerniniSemanticPlanner
from .renderer import MLPFlowRenderer
from .flow_matching import FlowMatchScheduler
from ..utils.seed import set_seed

class BerniniMNISTPipeline(nn.Module):
    """
    End-to-end Bernini MNIST Generator Pipeline:
    Class Label -> Semantic Planner (Qwen3-0.6B + Flow Matching) -> Refined Semantics (16x256) -> Pixel Denoiser (ResMLP) -> 28x28 Image.
    """
    def __init__(
        self,
        planner: BerniniSemanticPlanner,
        renderer: MLPFlowRenderer,
        scheduler: Optional[FlowMatchScheduler] = None
    ):
        super().__init__()
        self.planner = planner
        self.renderer = renderer
        self.scheduler = scheduler or FlowMatchScheduler()

    @classmethod
    def from_pretrained(
        cls,
        planner_path: Optional[str] = None,
        renderer_path: Optional[str] = None,
        mllm_model_name: str = "Qwen/Qwen3-0.6B",
        semantic_dim: int = 256,
        renderer_hidden: int = 128,
        renderer_layers: int = 4,
        planner_head_hidden: int = 512,
        planner_head_layers: int = 3,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> "BerniniMNISTPipeline":
        """
        Instantiate pipeline from checkpoint files.
        """
        planner = BerniniSemanticPlanner(
            mllm_model_name=mllm_model_name,
            semantic_dim=semantic_dim,
            num_semantic_tokens=16,
            num_classes=10,
            head_hidden_size=planner_head_hidden,
            head_layers=planner_head_layers,
            freeze_mllm=True
        ).to(device)

        if planner_path and os.path.exists(planner_path):
            print(f"Loading planner weights from {planner_path}...")
            ckpt = torch.load(planner_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            planner.load_state_dict(state_dict, strict=False)

        renderer = MLPFlowRenderer(
            in_dim=784,
            hidden_size=renderer_hidden,
            num_layers=renderer_layers,
            semantic_dim=semantic_dim,
            num_semantic_tokens=16,
            cond_type="semantic"
        ).to(device)

        if renderer_path and os.path.exists(renderer_path):
            print(f"Loading renderer weights from {renderer_path}...")
            ckpt = torch.load(renderer_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            renderer.load_state_dict(state_dict)

        return cls(planner=planner, renderer=renderer).to(device)

    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        num_refinement_steps: int = 4,
        ode_steps_planner: int = 20,
        ode_steps_renderer: int = 30,
        temperature: float = 1.0,
        cfg_scale: float = 3.5,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 28x28 MNIST images for given class labels.
        
        Args:
            class_labels: Tensor of shape (B,) with integer classes 0-9
            num_refinement_steps: K steps for masked token refinement (default 4)
            ode_steps_planner: Euler ODE steps for planner FM head
            ode_steps_renderer: Euler ODE steps for pixel renderer
            temperature: Prior noise scaling
            cfg_scale: Classifier-Free Guidance scale (default 3.5)
            seed: Optional random seed for reproducible sampling
            
        Returns:
            images: Tensor of shape (B, 1, 28, 28) in range [0, 1]
            semantic_tokens: Generated continuous semantic representations (B, 16, 256)
        """
        if seed is not None:
            set_seed(seed)

        self.planner.eval()
        self.renderer.eval()

        device = next(self.renderer.parameters()).device
        if not isinstance(class_labels, torch.Tensor):
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        class_labels = class_labels.to(device)

        # Stage 1: Generate refined semantic tokens with CFG
        semantic_tokens = self.planner.sample_iterative(
            class_labels=class_labels,
            num_refinement_steps=num_refinement_steps,
            ode_steps_per_refinement=ode_steps_planner,
            temperature=temperature,
            cfg_scale=cfg_scale
        )

        # Stage 2: Render continuous pixel image
        def velocity_fn(x, t):
            return self.renderer(x, t, semantic_tokens)

        sampled_x = self.scheduler.sample_euler(
            velocity_fn=velocity_fn,
            shape=(class_labels.shape[0], 784),
            device=device,
            steps=ode_steps_renderer
        )

        # Reshape to (B, 1, 28, 28) and unnormalize [-1, 1] -> [0, 1]
        images = (sampled_x.view(-1, 1, 28, 28) + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)

        return images, semantic_tokens
