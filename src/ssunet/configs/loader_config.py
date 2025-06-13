"""Training script."""

from dataclasses import dataclass

import torch.utils.data as dt


@dataclass
class LoaderConfig:
    """Data loader configuration."""

    batch_size: int = 20
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = True
    num_workers: int = 6
    persistent_workers: bool = True
    prefetch_factor: int | None = None

    @property
    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def name(self) -> str:
        """Get the name of the dataloader."""
        name_str = [
            f"bs={self.batch_size}",
            f"sh={self.shuffle}",
            f"pm={self.pin_memory}",
            f"dl={self.drop_last}",
            f"nw={self.num_workers}",
            f"pw={self.persistent_workers}",
        ]
        return "_".join(name for name in name_str if name is not None and name != "")

    def loader(self, data: dt.Dataset) -> dt.DataLoader:
        """Create a data loader with CUDA stream support.

        :param data: Dataset to load
        :type data: dt.Dataset
        :return: DataLoader configured with CUDA streaming
        :rtype: dt.DataLoader
        """
        return dt.DataLoader(data, **self.to_dict)
