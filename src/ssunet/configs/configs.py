"""Configuration for the project."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy
from typing import Any

import pytorch_lightning as pl
import torch

from ..constants import DEFAULT_CONFIG_PATH
from ..utils import _load_yaml
from .data_config import DataConfig
from .file_config import FileSource, PathConfig
from .loader_config import LoaderConfig
from .model_config import ModelConfig
from .split_config import SplitParams
from .train_config import TrainConfig


@dataclass
class MasterConfig:
    """Configuration class containing all configurations.

    :param data_config: Data configuration
    :param path_config: Path configuration
    :param split_params: Data split parameters
    :param model_config: Model configuration
    :param loader_config: Loader configuration
    :param train_config: Training configuration
    """

    data_config: DataConfig
    path_config: PathConfig
    split_params: SplitParams
    model_config: ModelConfig
    loader_config: LoaderConfig
    train_config: TrainConfig

    target_path: Path = field(init=False)
    time_stamp: str = field(init=False)

    def __post_init__(self) -> None:
        """Post initialization function.

        :returns: None
        """
        self.time_stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    @property
    def name(self) -> str:
        """Generate a name for the experiment.

        :returns: Experiment name string
        """
        name = "_".join(
            [
                self.time_stamp,
                self.train_config.name,
                self.data_config.name,
                self.model_config.name,
            ]
        )
        return name

    @property
    def device(self) -> torch.device:
        """Get the primary computation device based on trainer config.

        :returns: torch.device
        """
        accelerator: str | Any = self.train_config.accelerator
        devices_cfg: Any = self.train_config.devices

        if accelerator == "cuda":
            device_id: int
            if isinstance(devices_cfg, list):
                device_id = devices_cfg[0] if devices_cfg else -1
            else:
                device_id = devices_cfg
            if device_id == -1:
                return torch.device("cpu")
            return torch.device(f"cuda:{device_id}")
        return torch.device(accelerator if isinstance(accelerator, str) else "cpu")

    @property
    def data_path(self) -> Path:
        """Get the path to the data.

        :returns: Path to data file
        """
        return self.path_config.data_file

    @property
    def model_path(self) -> Path:
        """Get the path to the model.

        :returns: Path to model directory
        """
        return self.train_config.default_root_dir

    @property
    def log_path(self) -> Path:
        """Get the path to the log.

        :returns: Path to log directory
        """
        return self.train_config.default_root_dir / "logs"

    @property
    def checkpoint_path(self) -> Path:
        """Get the path to the checkpoint.

        :returns: Path to checkpoint file
        """
        return self.train_config.default_root_dir / "model.ckpt"

    @property
    def trainer(self) -> pl.Trainer:
        """Alias for the trainer.

        :returns: pl.Trainer instance
        """
        return self.train_config.trainer

    def copy_config(self, source_path: Path | str, target_path: Path | str | None = None) -> None:
        """Copy the configuration to a new directory.

        :param source_path: Source config file path
        :param target_path: Target directory path (optional)
        :returns: None
        """
        source_path = Path(source_path)
        source_file_name = source_path.name
        target_path = Path(target_path) if target_path is not None else self.target_path
        target_path.mkdir(parents=True, exist_ok=True)
        copy(source_path, target_path / source_file_name)

    @classmethod
    def from_config(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "MasterConfig":
        """Convert the configuration dictionary to dataclasses.

        :param config_path: Path to config YAML file
        :returns: MasterConfig instance
        """
        config_path = Path(config_path)
        config_dict = _load_yaml(config_path)

        path_section = config_dict.get("PATH", {})

        data_source_dict = path_section.get("data", {})
        data_fsc = FileSource(**data_source_dict) if data_source_dict else FileSource()

        reference_source_dict = path_section.get("reference")
        reference_fsc = FileSource(**reference_source_dict) if reference_source_dict else None

        ground_truth_source_dict = path_section.get("ground_truth")
        ground_truth_fsc = (
            FileSource(**ground_truth_source_dict) if ground_truth_source_dict else None
        )

        path_cfg = PathConfig(
            base_dir=path_section.get("base_dir"),
            data=data_fsc,
            reference=reference_fsc,
            ground_truth=ground_truth_fsc,
        )

        master_config = MasterConfig(
            path_config=path_cfg,
            data_config=DataConfig(**config_dict["DATA"]),
            split_params=SplitParams(**config_dict["SPLIT"]),
            model_config=ModelConfig(**config_dict["MODEL"]),
            loader_config=LoaderConfig(**config_dict["LOADER"]),
            train_config=TrainConfig(**config_dict["TRAIN"]),
        )
        if len(config_path.parent.name) >= len(master_config.name):
            master_config.time_stamp = master_config.name[:15]
            master_config.train_config.set_new_root(new_root=config_path.parent)
            master_config.target_path = config_path.parent
        else:
            master_config.train_config.set_new_root(new_root=master_config.name)
            master_config.target_path = master_config.train_config.default_root_dir
        return master_config
