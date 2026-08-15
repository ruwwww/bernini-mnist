import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Tuple, Callable

from .flow_matching import FlowMatchScheduler
from .renderer import TimestepEmbedder, modulate

class SemanticAdaLNResBlock(nn.Module):
    """Pointwise MLP Residual Block modulated by AdaLN (Adaptive Layer Normalization)."""
    def __init__(self, hidden_size: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size)  # shift, scale, gate
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h

class SemanticFlowMatchingHead(nn.Module):
    """
    Decoder Head that predicts continuous target semantic velocity v(z_t, t, H)
    conditioned on Qwen contextual hidden states H.
    """
    def __init__(
        self,
        semantic_dim: int = 256,
        qwen_hidden_size: int = 1024,
        head_hidden_size: int = 512,
        num_layers: int = 3
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.input_proj = nn.Linear(semantic_dim, head_hidden_size)
        self.time_embed = TimestepEmbedder(head_hidden_size, frequency_embedding_size=256)
        self.cond_proj = nn.Linear(qwen_hidden_size, head_hidden_size)

        self.blocks = nn.ModuleList([
            SemanticAdaLNResBlock(head_hidden_size) for _ in range(num_layers)
        ])

        self.norm_final = nn.LayerNorm(head_hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(head_hidden_size, 2 * head_hidden_size)
        )
        self.output_proj = nn.Linear(head_hidden_size, semantic_dim)

        self._init_weights()

    def _init_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_final[-1].bias, 0)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        h_qwen: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_t: Noisy semantic latents (B, N, semantic_dim)
            t: Timesteps (B,)
            h_qwen: Qwen hidden states at semantic token positions (B, N, qwen_hidden_size)
        Returns:
            v_pred: Predicted velocity (B, N, semantic_dim)
        """
        x = self.input_proj(z_t)                     # (B, N, head_hidden)
        t_emb = self.time_embed(t).unsqueeze(1)      # (B, 1, head_hidden)
        c_emb = self.cond_proj(h_qwen)               # (B, N, head_hidden)

        c = t_emb + c_emb                            # (B, N, head_hidden)

        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.adaLN_final(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        v_pred = self.output_proj(x)
        return v_pred

class LoRALinear(nn.Module):
    """Native PyTorch Low-Rank Adaptation (LoRA) layer for attention projections."""
    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False
        self.scaling = lora_alpha / r
        self.lora_A = nn.Linear(base_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + self.scaling * self.lora_B(self.lora_A(x))

class BerniniSemanticPlanner(nn.Module):
    """
    Stage 1 Semantic Planner using:
    1. Qwen3-0.6B with native PyTorch LoRA attention adaptation
    2. Classifier-Free Guidance (CFG) support
    3. MaskGIT progressive spatial unmasking
    4. Flow Matching MLP Head
    """
    def __init__(
        self,
        mllm_model_name: str = "Qwen/Qwen3-0.6B",
        semantic_dim: int = 256,
        num_semantic_tokens: int = 16,
        num_classes: int = 10,
        head_hidden_size: int = 512,
        head_layers: int = 3,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        freeze_mllm: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.num_semantic_tokens = num_semantic_tokens
        self.num_classes = num_classes
        self.null_class_idx = num_classes  # Class index 10 is the unconditional null token for CFG

        # Load Qwen backbone
        self.mllm_config = AutoConfig.from_pretrained(mllm_model_name)
        self.qwen_hidden_size = getattr(self.mllm_config, "hidden_size", 1024)

        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mllm = AutoModel.from_pretrained(mllm_model_name, torch_dtype=torch.float32)

        # Freeze base parameters
        for p in self.mllm.parameters():
            p.requires_grad = False

        if use_lora:
            num_lora_layers = 0
            for layer in self.mllm.layers:
                layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_r, lora_alpha=lora_alpha)
                layer.self_attn.k_proj = LoRALinear(layer.self_attn.k_proj, r=lora_r, lora_alpha=lora_alpha)
                layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_r, lora_alpha=lora_alpha)
                layer.self_attn.o_proj = LoRALinear(layer.self_attn.o_proj, r=lora_r, lora_alpha=lora_alpha)
                num_lora_layers += 4
            print(f"[BerniniPlanner] Applied native LoRA (r={lora_r}, alpha={lora_alpha}) across {num_lora_layers} attention modules.")

        # Class embedding table: (num_classes + 1, hidden_size) where index num_classes is null/uncond
        self.class_embed = nn.Embedding(num_classes + 1, self.qwen_hidden_size)

        # Learnable MASK tokens for the 16 positions (1, 16, hidden_size)
        self.mask_tokens = nn.Parameter(torch.zeros(1, num_semantic_tokens, self.qwen_hidden_size))

        # Linear projection from continuous ViT dimension to Qwen hidden dimension
        self.vit_proj = nn.Linear(semantic_dim, self.qwen_hidden_size)

        # Flow Matching MLP Decoder Head
        self.fm_head = SemanticFlowMatchingHead(
            semantic_dim=semantic_dim,
            qwen_hidden_size=self.qwen_hidden_size,
            head_hidden_size=head_hidden_size,
            num_layers=head_layers
        )

        self.scheduler = FlowMatchScheduler()
        self._init_weights()
        self.to(target_device)

    def _init_weights(self):
        nn.init.trunc_normal_(self.class_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.mask_tokens, std=0.02)
        nn.init.xavier_uniform_(self.vit_proj.weight)
        if self.vit_proj.bias is not None:
            nn.init.constant_(self.vit_proj.bias, 0)

    def get_qwen_hidden_states(
        self,
        class_labels: torch.Tensor,
        z_semantic: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Construct sequence [ClassToken, E_1, ..., E_16] and forward through Qwen.
        """
        B = class_labels.shape[0]
        target_device = next(self.class_embed.parameters()).device
        class_labels = class_labels.to(target_device)

        # Class embedding: (B, 1, hidden_size)
        cls_tokens = self.class_embed(class_labels).unsqueeze(1)

        # Base mask tokens repeated across batch: (B, 16, hidden_size)
        mask_embeds = self.mask_tokens.expand(B, -1, -1)

        if z_semantic is not None and mask is not None:
            z_semantic = z_semantic.to(target_device)
            mask = mask.to(target_device)
            sem_proj = self.vit_proj(z_semantic)
            mask_expanded = mask.unsqueeze(-1).to(sem_proj.dtype)  # 1 for masked, 0 for committed
            token_embeds = mask_expanded * mask_embeds + (1.0 - mask_expanded) * sem_proj
        else:
            token_embeds = mask_embeds

        inputs_embeds = torch.cat([cls_tokens, token_embeds], dim=1)

        outputs = self.mllm(inputs_embeds=inputs_embeds)
        last_hidden = outputs.last_hidden_state

        sem_hidden = last_hidden[:, 1:, :]  # (B, 16, hidden_size)
        return sem_hidden

    def compute_loss(
        self,
        class_labels: torch.Tensor,
        z_1: torch.Tensor,
        mask_ratio: float = 0.5,
        cfg_drop_rate: float = 0.10
    ) -> torch.Tensor:
        """
        Compute Flow Matching loss with Classifier-Free Guidance (CFG) random class dropout.
        """
        B, N, D = z_1.shape
        target_device = next(self.class_embed.parameters()).device
        class_labels = class_labels.to(target_device)
        z_1 = z_1.to(target_device)

        # CFG class dropout: drop class label to null token 10% of the time
        if cfg_drop_rate > 0:
            drop_mask = torch.rand(B, device=target_device) < cfg_drop_rate
            effective_labels = torch.where(
                drop_mask,
                torch.tensor(self.null_class_idx, device=target_device),
                class_labels
            )
        else:
            effective_labels = class_labels

        # Random binary mask for MaskGIT training: 1 = masked (predict), 0 = unmasked (context)
        rand = torch.rand(B, N, device=target_device)
        mask = (rand < mask_ratio).float()
        mask[:, 0] = torch.where(mask.sum(dim=1) == 0, torch.tensor(1.0, device=target_device), mask[:, 0])

        # Qwen forward pass with LoRA
        h_qwen = self.get_qwen_hidden_states(effective_labels, z_semantic=z_1, mask=mask)

        # Flow matching loss on masked continuous latents
        z_t, target_v, t = self.scheduler.add_noise(z_1)
        pred_v = self.fm_head(z_t, t, h_qwen)

        loss = self.scheduler.compute_loss(pred_v, target_v, mask=mask)
        return loss

    @torch.no_grad()
    def sample_iterative(
        self,
        class_labels: torch.Tensor,
        num_refinement_steps: int = 4,
        ode_steps_per_refinement: int = 20,
        temperature: float = 1.0,
        cfg_scale: float = 3.5
    ) -> torch.Tensor:
        """
        Classifier-Free Guided MaskGIT Progressive Refinement Sampling.
        
        Applies CFG:
            v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        across the iterative unmasking schedule.
        """
        target_device = next(self.class_embed.parameters()).device
        class_labels = class_labels.to(target_device)
        B = class_labels.shape[0]
        N = self.num_semantic_tokens
        D = self.semantic_dim

        null_labels = torch.full((B,), self.null_class_idx, dtype=torch.long, device=target_device)

        current_z = torch.zeros(B, N, D, device=target_device)
        current_mask = torch.ones(B, N, device=target_device)  # 1 = masked, 0 = committed

        # Spatial priority for progressive unmasking: center body first, then outer edges
        spatial_priority = torch.tensor([5, 6, 9, 10, 1, 2, 4, 7, 8, 11, 13, 14, 0, 3, 12, 15], device=target_device)

        for step in range(num_refinement_steps):
            # Compute conditional & unconditional Qwen hidden states
            h_cond = self.get_qwen_hidden_states(class_labels, z_semantic=current_z, mask=current_mask)
            
            if cfg_scale > 1.0:
                h_uncond = self.get_qwen_hidden_states(null_labels, z_semantic=current_z, mask=current_mask)
                def velocity_fn(z_val, t_val):
                    v_c = self.fm_head(z_val, t_val, h_cond)
                    v_u = self.fm_head(z_val, t_val, h_uncond)
                    return v_u + cfg_scale * (v_c - v_u)
            else:
                def velocity_fn(z_val, t_val):
                    return self.fm_head(z_val, t_val, h_cond)

            z_0 = torch.randn(B, N, D, device=target_device) * temperature
            candidate_z = self.scheduler.sample_euler(
                velocity_fn,
                shape=(B, N, D),
                device=target_device,
                steps=ode_steps_per_refinement,
                x_0=z_0
            )

            # Update candidate tokens into masked positions
            mask_exp = current_mask.unsqueeze(-1)
            current_z = (1.0 - mask_exp) * current_z + mask_exp * candidate_z

            if step < num_refinement_steps - 1:
                # MaskGIT cosine schedule: fraction of tokens to remain masked
                mask_ratio = math.cos(math.pi * 0.5 * (step + 1) / float(num_refinement_steps))
                num_to_mask = max(1, int(math.ceil(N * mask_ratio)))
                num_to_commit = N - num_to_mask

                new_mask = torch.ones(B, N, device=target_device)
                for b in range(B):
                    commit_indices = spatial_priority[:num_to_commit]
                    new_mask[b, commit_indices] = 0.0
                current_mask = new_mask
            else:
                current_mask = torch.zeros(B, N, device=target_device)

        return current_z
