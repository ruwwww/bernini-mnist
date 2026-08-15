import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional

class FlowMatchScheduler:
    """
    Optimal Transport Flow Matching Scheduler.
    
    Probability path:
        x_t = (1 - t) * x_0 + t * x_1,  where x_0 ~ N(0, I), x_1 ~ data distribution.
    Target velocity:
        v_t = x_1 - x_0
    """
    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample continuous timesteps uniformly from [0, 1]."""
        return torch.rand(batch_size, device=device)

    def add_noise(
        self,
        x_1: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interpolate between noise x_0 and data x_1 at timestep t.
        
        Args:
            x_1: Target data tensor of shape (B, ...)
            x_0: Optional noise tensor of same shape as x_1. If None, sampled from N(0, I)
            t: Optional timestep tensor of shape (B,). If None, sampled uniformly.
            
        Returns:
            x_t: Interpolated noisy tensor at timestep t
            target_velocity: Ground truth velocity vector field (x_1 - x_0)
            t: Timesteps used
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        if t is None:
            t = self.sample_timesteps(x_1.shape[0], x_1.device)
            
        # Broadcast t across remaining dimensions (e.g., (B, 1) or (B, 1, 1))
        t_expand = t.view(x_1.shape[0], *([1] * (x_1.dim() - 1)))
        
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        target_velocity = x_1 - x_0
        return x_t, target_velocity, t

    def compute_loss(
        self,
        model_pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute mean squared error loss between predicted velocity and target velocity.
        """
        loss = F.mse_loss(model_pred_velocity, target_velocity, reduction="none")
        if mask is not None:
            # mask shape (B, N) or (B, N, 1)
            if mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)
        return loss.mean()

    @torch.no_grad()
    def sample_euler(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 50,
        x_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Numerically solve ODE dx/dt = v_theta(x_t, t) from t=0 to t=1 using Euler steps.
        
        Args:
            velocity_fn: Callable (x_t, t) -> v_pred of same shape as x_t.
            shape: Shape of the tensor to generate, e.g. (B, 784) or (B, 16, 256).
            device: Target device.
            steps: Number of Euler integration steps.
            x_0: Optional starting noise. If None, sampled from N(0, I).
            
        Returns:
            x_1: Final generated tensor at t=1.
        """
        if x_0 is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = x_0.clone()

        dt = 1.0 / steps
        for step in range(steps):
            t_val = step / steps
            t = torch.full((shape[0],), t_val, device=device, dtype=x_t.dtype)
            v = velocity_fn(x_t, t)
            x_t = x_t + dt * v

        return x_t
