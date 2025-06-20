import math

import torch
import torch.nn.functional as tnf
from torch import nn


# Custom Activation Functions
class GatedReLUMix(nn.Module):
    """Mixes two parallel processing paths using ReLU and gating mechanism."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels % 2 != 0:
            raise ValueError(f"Input channels must be even for GatedReLUMix, but got {channels}")
        x1, x2 = torch.chunk(x, 2, dim=1)
        out1 = tnf.relu(x1)
        out2 = x1 * x2
        return torch.cat((out1, out2), dim=1)


class SinGatedMix(nn.Module):
    """Mixes a direct sinusoidal activation with sinusoidally gated second half."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels % 2 != 0:
            raise ValueError(f"Input channels must be even for SinGatedMix, but got {channels}")
        x1, x2 = torch.chunk(x, 2, dim=1)
        sin_gate_x1 = torch.sin(x1)
        out1 = sin_gate_x1
        out2 = sin_gate_x1 * x2
        return torch.cat((out1, out2), dim=1)


# Gating Mechanism (used by NAFNet)
class SimpleGate3D(nn.Module):
    """3D Simple Gate mechanism from NAFNet (element-wise multiply after channel chunk)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Input channels ({x.shape[1]}) must be even for SimpleGate3D.")
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# Scaled Simple Gate (to prevent numerical overflow)
class ScaledSimpleGate3D(nn.Module):
    """Scaled Simple Gate operation for 3D data to prevent numerical overflow."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Split channels in half
        x1, x2 = x.chunk(2, dim=1)
        # Scaled element-wise multiplication
        return x1 * x2 / math.sqrt(self.dim)
