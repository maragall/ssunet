import torch
import torch.nn as nn
import torch.nn.functional as tnf


class GatedReLUMix(nn.Module):
    """Mixes two parallel processing paths, maintaining channel dimension.

    The module processes input features (tensor 'x') through two parallel paths
    and concatenates the results along the channel dimension to maintain the
    input channel dimension (2*C). This process combines ReLU with a simple
    element-wise gate.

    - Path 1: Applies ReLU to the first half of the input channels (x1).
    - Path 2: Implements a SimpleGate mechanism that uses x1 to gate x2
              (element-wise multiplication: x1 * x2).

    The SimpleGate mechanism, in this case, facilitates controlled information
    flow through the network, wherein x1 determines which features in x2 are
    passed through.
    This module expects an input tensor with an even number of channels.
    Output tensor will have the same number of channels as the input (2*C).
    """

    def __init__(self) -> None:
        """Initialize the GatedReLUMix module."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for GatedReLUMix.

        :param x: Input tensor. Expected to have an even number of channels (2*C), with a
            shape of (N, 2C, *), where N is the batch size, 2C is the number of
            input channels, and * represents any additional dimensions.
        :raises ValueError: If the input tensor does not have an even number of channels.
        :return: Output tensor with the same number of channels as the input (2*C),
                 shaped (N, 2C, *).
        """
        channels: int = x.shape[1]
        if channels % 2 != 0:
            raise ValueError(f"Input channels must be even for GatedReLUMix, but got {channels}")

        # Split the input tensor into two halves along the channel dimension
        x1: torch.Tensor
        x2: torch.Tensor
        x1, x2 = torch.chunk(x, 2, dim=1)  # Each part has C channels

        # Path 1: Apply ReLU to the first half
        out1: torch.Tensor = tnf.relu(x1, inplace=True)

        # Path 2: Apply SimpleGate using both halves
        out2: torch.Tensor = x1 * x2

        # Concatenate the results along the channel dimension
        output: torch.Tensor = torch.cat((out1, out2), dim=1)

        return output


class SinGatedReLUMix(nn.Module):
    """Mixes ReLU on the first half of channels with a sinusoidally gated version.

    Path 1: Applies ReLU to the first half of the channels (x1).
    Path 2: Applies sin(x1) * x2. Here, sin(x1) acts as a sinusoidal gate for x2.

    Input tensor is expected to have an even number of channels.
    Output tensor has the same number of channels as the input.
    """

    def __init__(self) -> None:
        """Initialize the SinGatedReLUMix module."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SinGatedReLUMix.

        :param x: Input tensor. Expected to have an even number of channels (2*C), with a
            shape of (N, 2C, *), where N is the batch size, 2C is the number of
            input channels, and * represents any additional dimensions.
        :raises ValueError: If the input tensor does not have an even number of channels.
        :return: Output tensor with the same number of channels as the input (2*C),
                 shaped (N, 2C, *).
        """
        channels = x.shape[1]  # Type 'int' is obvious
        if channels % 2 != 0:
            raise ValueError(f"Input channels must be even for SinGatedReLUMix, but got {channels}")

        # Split the input tensor into two halves along the channel dimension
        # Types 'torch.Tensor' for x1 and x2 are obvious from torch.chunk
        x1, x2 = torch.chunk(x, 2, dim=1)

        # Path 1: Apply ReLU to the first half
        out1 = tnf.relu(x1)
        out2 = torch.sin(x1) * x2

        # Concatenate the results
        output = torch.cat((out1, out2), dim=1)
        return output


class SinGatedMix(nn.Module):
    """Mixes a direct sinusoidal activation on the first half of channels with a sinusoidally gated.

    Path 1: Applies sin(x1) to the first half of the channels.
    Path 2: Applies sin(x1) * x2. Here, sin(x1) acts as a sinusoidal gate for x2.

    Input tensor is expected to have an even number of channels.
    Output tensor has the same number of channels as the input.
    """

    def __init__(self) -> None:
        """Initialize the SinGatedMix module."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SinGatedMix.

        :param x: Input tensor. Expected to have an even number of channels (2*C), with a
            shape of (N, 2C, *), where N is the batch size, 2C is the number of
            input channels, and * represents any additional dimensions.
        :raises ValueError: If the input tensor does not have an even number of channels.
        :return: Output tensor with the same number of channels as the input (2*C),
                 shaped (N, 2C, *).
        """
        channels = x.shape[1]
        if channels % 2 != 0:
            raise ValueError(f"Input channels must be even for SinGatedMix, but got {channels}")

        # Split the input tensor
        x1, x2 = torch.chunk(x, 2, dim=1)
        sin_gate_x1 = torch.sin(x1)

        out1 = sin_gate_x1
        out2 = sin_gate_x1 * x2

        # Path 2: Apply Sinusoidal Gate
        out2 = sin_gate_x1 * x2  # Type obvious

        # Concatenate the results
        output = torch.cat((out1, out2), dim=1)  # Type obvious
        return output
