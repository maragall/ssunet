from dataclasses import dataclass

from .options import (
    ActivationNameOptions,
    DownModeOptions,
    MergeModeOptions,
    NormTypeOptions,
    UpModeOptions,
)


@dataclass
class BaseUnetBlock3DConfig:
    in_channels: int
    out_channels: int
    skip_out: bool = True
    dropout_p: float = 0.0
    down_mode: DownModeOptions = "maxpool"
    up_mode: UpModeOptions = "transpose"
    merge_mode: MergeModeOptions = "concat"
    z_conv: bool = True
    activation_name: ActivationNameOptions = "gelu"
    activation_params: dict | None = None
    norm_type: NormTypeOptions = "layer"
    norm_params: dict | None = None
    num_blocks: int = 2
    intermediate_channels_list: list[int] | None = None
    use_depthwise_separable_conv: bool = False

    def __post_init__(self):
        if self.norm_params is None:
            self.norm_params = {}
