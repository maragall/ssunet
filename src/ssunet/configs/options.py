from typing import Literal

DownModeOptions = Literal["maxpool", "avgpool", "conv", "unshuffle"]
UpModeOptions = Literal["transpose", "upsample", "pixelshuffle"]
MergeModeOptions = Literal["concat", "add"]
ActivationNameOptions = Literal[
    "relu",
    "leakyrelu",
    "prelu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "softmax",
    "logsoftmax",
]
NormTypeOptions = Literal["layer", "batch", "group", "instance"]
LossFunctionOptions = Literal["mse", "l1", "photon", "photon_2d"]
BlockTypeOptions = Literal["efficient", "convnext", "nafnet", "tri"]
SplitMethodOptions = Literal["signal", "fixed", "list"]
LRSchedulerOptions = Literal[
    "cosine_annealing", "cosine_annealing_warm_restarts", "reduce_lr_on_plateau", "none"
]
OptimizerNameOptions = Literal["adam", "sgd", "adamw"]
PrecisionOptions = Literal["64", "32", "16", "bf16", "mixed"]
AcceleratorOptions = Literal["cpu", "cuda", "mps", "ipu", "tpu", "auto"]
MonitorModeOptions = Literal["min", "max"]
LoggingIntervalOptions = Literal["step", "epoch"]
