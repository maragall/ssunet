# src/ssunet/configs/split_config.py
from dataclasses import dataclass, field

from ..constants import EPSILON
from ..exceptions import ConfigError


@dataclass
class SplitParams:
    method: str = "signal"
    min_p: float = EPSILON
    max_p: float = 1.0 - EPSILON
    p_list: list[float] | None = field(default_factory=list)
    normalize_target: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        valid_methods = ["signal", "fixed", "list"]
        if self.method not in valid_methods:
            raise ConfigError(
                f"Invalid SPLIT.method '{self.method}'. Must be one of {valid_methods}."
            )
        if self.method == "signal":
            if not (0.0 <= self.min_p < self.max_p <= 1.0):
                if not (
                    self.min_p == self.max_p and 0.0 <= self.min_p <= 1.0
                ):  # Allow min_p == max_p
                    raise ConfigError(
                        f"For 'signal' method, "
                        f"0.0 <= min_p ({self.min_p}) < max_p ({self.max_p}) <= 1.0, "
                        "or min_p == max_p within [0,1], must hold."
                    )
        elif self.method == "list" and self.p_list:
            for p_val in self.p_list:
                if not (0.0 < p_val < 1.0):  # p values in list should be (0,1) for binomial prob
                    raise ConfigError(
                        f"p_list values must be between 0 and 1 (exclusive), got {p_val}"
                    )
