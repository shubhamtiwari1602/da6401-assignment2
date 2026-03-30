"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        
        # Inverted dropout:
        # 1. Generate binomial mask with probability (1 - p)
        # 2. Scale by 1 / (1 - p) to preserve expected value
        scale = 1.0 / (1.0 - self.p)
        mask = (torch.rand(x.shape, device=x.device, dtype=x.dtype) > self.p).float()
        return x * mask * scale
