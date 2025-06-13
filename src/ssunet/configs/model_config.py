"""Configuration for the project."""

from dataclasses import dataclass, field
from typing import Any

from .options import (
    ActivationNameOptions,
    BlockTypeOptions,
    DownModeOptions,
    LossFunctionOptions,
    MergeModeOptions,
    UpModeOptions,
)


@dataclass
class ModelConfig:
    """Configuration for the SSUnet model."""

    channels: int = 1
    depth: int = 5
    start_filts: int = 32
    depth_scale: int = 2
    depth_scale_stop: int = 10
    z_conv_stage: int = 3
    group_norm: int = 8
    skip_depth: int = 0
    dropout_p: float = 0.0
    scale_factor: float = 10.0
    masked: bool = True
    down_checkpointing: bool = True
    up_checkpointing: bool = False
    loss_function: LossFunctionOptions = "photon"
    up_mode: UpModeOptions = "pixelshuffle"
    merge_mode: MergeModeOptions = "concat"
    down_mode: DownModeOptions = "maxpool"
    activation: ActivationNameOptions = "gelu"
    block_type: BlockTypeOptions = "tri"
    note: str = ""
    optimizer_config: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Generate the name of the model."""
        name_str = [
            f"d={self.depth}",
            f"sf={self.start_filts}",
            f"ds={self.depth_scale}at{self.depth_scale_stop}",
            f"f={self.scale_factor}",
            f"z={self.z_conv_stage}",
            f"g={self.group_norm}",
            f"sd={self.skip_depth}",
            f"b={self.block_type}",
            f"a={self.activation}",
        ]
        return "_".join(name_str)
