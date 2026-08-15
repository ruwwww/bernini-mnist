import torch
import pytest
from bernini_mnist.models.flow_matching import FlowMatchScheduler

def test_flow_match_scheduler_add_noise():
    scheduler = FlowMatchScheduler()
    x_1 = torch.randn(4, 784)
    x_0 = torch.randn(4, 784)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    
    x_t, target_v, sampled_t = scheduler.add_noise(x_1, x_0=x_0, t=t)
    
    assert x_t.shape == (4, 784)
    assert target_v.shape == (4, 784)
    # At t=0, x_t should equal x_0
    assert torch.allclose(x_t[0], x_0[0], atol=1e-6)
    # At t=1, x_t should equal x_1
    assert torch.allclose(x_t[3], x_1[3], atol=1e-6)
    # Target velocity should be x_1 - x_0
    assert torch.allclose(target_v, x_1 - x_0, atol=1e-6)

def test_flow_match_euler_sampling():
    scheduler = FlowMatchScheduler()
    # Dummy constant velocity function pointing towards target ones
    target = torch.ones(2, 784)
    def dummy_velocity(x_t, t):
        return target - x_t  # dx/dt = 1 - x => x(1) -> 1 - e^-1 ~ 0.63
    
    x_sampled = scheduler.sample_euler(
        dummy_velocity,
        shape=(2, 784),
        device=torch.device("cpu"),
        steps=20,
        x_0=torch.zeros(2, 784)
    )
    assert x_sampled.shape == (2, 784)
    assert not torch.isnan(x_sampled).any()
