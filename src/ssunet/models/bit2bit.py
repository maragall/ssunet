"""SSUnet model."""

import pytorch_lightning as pl
import torch
import torchmetrics.image as tmi
from torch import nn
from torch.nn import init
from torch.utils.checkpoint import checkpoint

from ..configs.configs import ModelConfig
from ..constants import EPSILON, LOGGER
from ..exceptions import InvalidUpModeError
from ..losses import loss_functions
from ..modules import BLOCK, conv111

OPTIMIZER = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
}


class Bit2Bit(pl.LightningModule):
    """Bit2Bit model."""

    def __init__(
        self,
        config: ModelConfig,
        **kwargs,
    ) -> None:
        """Initialize the Bit2Bit model.

        :param config: configuration for the model
        :param loss_function: loss function
        :param psnr_metric: peak signal-to-noise ratio metric
        :param ssim_metric: structural similarity index metric
        :param kwargs: additional arguments
        """
        super().__init__()

        # Replace pyiqa metrics with torchmetrics
        self.psnr_metric = tmi.PeakSignalNoiseRatio()
        self.ssim_metric = tmi.StructuralSimilarityIndexMeasure()

        self.config = config
        self.loss_function = loss_functions[config.loss_function]
        self.kwargs = kwargs
        self._check_conflicts()

        self.down_convs = self._down_conv_list()
        self.up_convs = self._up_conv_list()
        self.conv_final = self._final_conv()

        self.save_hyperparameters()  # Ensure this uses the updated ModelConfig
        self._reset_params()

    def _down_conv_list(self) -> nn.ModuleList:
        """Generate the list of down convolutional layers."""
        down_convs = []
        down_conv_function = BLOCK[self.config.block_type][0]
        initial_channels = self.config.channels

        for i in range(self.config.depth):
            z_conv = i < self.config.z_conv_stage
            skip_out = i >= self.config.skip_depth
            in_channels = (
                initial_channels
                if i == 0
                else self.config.start_filts * (self.config.depth_scale ** (i - 1))
            )
            out_channels = self.config.start_filts * (self.config.depth_scale**i)
            last = True if i == self.config.depth - 1 else False
            down_conv = down_conv_function(
                int(in_channels),
                int(out_channels),
                last=last,
                skip_out=skip_out,
                z_conv=z_conv,
                dropout_p=self.config.dropout_p,
                group_norm=self.config.group_norm,
                down_mode=self.config.down_mode,
                activation=self.config.activation,
            )
            down_convs.append(down_conv)
        return nn.ModuleList(down_convs)

    def _up_conv_list(self) -> nn.ModuleList:
        """Generate the list of up convolutional layers."""
        up_convs = []
        up_conv_function = BLOCK[self.config.block_type][1]
        for i in range(self.config.depth - 1, 0, -1):
            z_conv = (i - 1) < self.config.z_conv_stage
            skip_out = i >= self.config.skip_depth
            in_channels = self.config.start_filts * (self.config.depth_scale**i)
            out_channels = self.config.start_filts * (self.config.depth_scale ** (i - 1))
            up_conv = up_conv_function(
                int(in_channels),
                int(out_channels),
                z_conv=z_conv,
                skip_out=skip_out,
                dropout_p=self.config.dropout_p,
                group_norm=self.config.group_norm,
                up_mode=self.config.up_mode,
                activation=self.config.activation,
            )
            up_convs.append(up_conv)
        return nn.ModuleList(up_convs)

    def _final_conv(self):
        """Generate the final convolutional layer."""
        return nn.Sequential(
            conv111(self.config.start_filts, self.config.channels),
        )

    @staticmethod
    def _weight_init(module: nn.Module) -> None:
        """Initialize the weights of the model."""
        if isinstance(module, nn.Conv3d):
            init.xavier_normal_(module.weight)
            init.constant_(module.bias, 0)  # type: ignore

    def _reset_params(self) -> None:
        """Reset the parameters of the model."""
        for module in self.modules():
            self._weight_init(module)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        encoder_outs = []
        for i, down_conv in enumerate(self.down_convs):
            if self.config.down_checkpointing:
                input, skip = checkpoint(down_conv, input, use_reentrant=False)  # type: ignore
            else:
                input, skip = down_conv(input)
            encoder_outs.append(skip) if i < self.config.depth - 1 else ...
            del skip

        for up_conv in self.up_convs:
            skip = encoder_outs.pop()
            if self.config.up_checkpointing:
                input = checkpoint(up_conv, input, skip, use_reentrant=False)  # type: ignore
            else:
                input = up_conv(input, skip)
        return self.conv_final(input)

    def configure_optimizers(self) -> dict:
        """Configure the optimizer and scheduler."""
        opt_config = self.config.optimizer_config  # This is a dict from ModelConfig

        optimizer = (
            OPTIMIZER[opt_config["name"]](self.parameters(), lr=opt_config["lr"], fused=False)
            if opt_config["name"] in ("adam", "adamw")
            and "fused" in torch.optim.AdamW.__init__.__kwdefaults__  # Check if fused is supported
            else OPTIMIZER[opt_config["name"]](self.parameters(), lr=opt_config["lr"])
        )

        scheduler_name = opt_config.get("scheduler_name", "reduce_lr_on_plateau").lower()

        if scheduler_name == "cosine_annealing":
            # Determine T_max: use configured value or trainer's max_epochs
            # self.trainer is available in configure_optimizers
            t_max_epochs = opt_config.get("cosine_t_max_epochs")
            if t_max_epochs is None:
                if (
                    self.trainer
                    and self.trainer.max_epochs is not None
                    and self.trainer.max_epochs != -1
                ):
                    t_max = self.trainer.max_epochs
                else:
                    # Fallback if trainer.max_epochs isn't available or set,
                    # though this should ideally come from config or be ensured by trainer setup.
                    # You might need to pass max_epochs from TrainConfig to ModelConfig
                    # if self.trainer isn't fully set up.
                    # For now, let's assume a default or raise an error.
                    raise ValueError(
                        "CosineAnnealingLR T_max could not be determined. "
                        "Ensure TRAIN.max_epochs is set or provide "
                        "'cosine_t_max_epochs' in optimizer_config."
                    )
            else:
                t_max = int(t_max_epochs)

            eta_min = float(opt_config.get("cosine_eta_min", 0.0))

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,  # Number of epochs or steps
                eta_min=eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Cosine annealing is typically updated per epoch
                    "frequency": 1,
                },
            }
        elif scheduler_name == "cosine_annealing_warm_restarts":
            t_0 = opt_config.get("cosine_restarts_t_0")
            if t_0 is None:
                raise ValueError(
                    "CosineAnnealingWarmRestarts 'cosine_restarts_t_0' "
                    "(epochs for the first restart) must be provided in optimizer_config."
                )
            t_0 = int(t_0)
            t_mult = int(opt_config.get("cosine_restarts_t_mult", 1))
            eta_min = float(opt_config.get("cosine_restarts_eta_min", 0.0))

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Typically updated per epoch
                    "frequency": 1,
                },
            }
        elif scheduler_name == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=opt_config.get("plateau_mode", "min"),
                factor=float(opt_config.get("plateau_factor", 0.5)),
                patience=int(opt_config.get("plateau_patience", 10)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config.optimizer_config.get(
                        "plateau_monitor", "val_loss"
                    ),  # Ensure monitor is configurable
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # No scheduler or unknown scheduler
            return {"optimizer": optimizer}

    def training_step(
        self,
        batch: list[torch.Tensor],  # batch of training data
        batch_idx,
    ) -> torch.Tensor:
        """Training step of the model."""
        y_loss_target = batch[0]
        x = batch[1]
        y_pred = self(x)
        loss = (
            self.loss_function(y_pred, y_loss_target, (x < 1).float())
            if self.config.masked
            else self.loss_function(y_pred, y_loss_target)
        )
        self._tb_train_log(loss, y_pred, y_loss_target, batch_idx)
        return loss

    def _tb_train_log(
        self,
        loss: torch.Tensor,
        prediction: torch.Tensor,
        loss_target: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log the training step."""
        self.log("train_loss", loss)
        self._log_image(prediction[0], "train_image", batch_idx, frequency=100)

    def validation_step(
        self,
        batch: list[torch.Tensor],  # batch of validation data
        batch_idx: int,
    ) -> None:
        """Validation step of the model."""
        y_loss_target = batch[0]
        x = batch[1]
        y_metrics_target = batch[2] if len(batch) == 3 else None
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_loss_target)
        self._tb_val_log(loss, y_pred, y_loss_target, y_metrics_target, batch_idx)

    def _tb_val_log(
        self,
        loss: torch.Tensor,
        prediction: torch.Tensor,
        loss_target: torch.Tensor,
        metrics_target: torch.Tensor | None,
        batch_idx: int,
    ) -> None:
        """Log the validation step. Can be extended for logging metrics."""
        if metrics_target is not None:
            self._log_metrics(prediction, metrics_target, batch_idx)
        self.log("val_loss", loss)
        self._log_image(prediction[0], "val_image", batch_idx, frequency=10)

    def _log_metrics(
        self,
        prediction: torch.Tensor,
        metrics_target: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log the metrics.

        :param prediction: Model output tensor
        :param metrics_target: Ground truth tensor for metrics
        :param batch_idx: Current batch index
        """
        size_z = metrics_target.shape[2]
        index_z = size_z // 2

        # Extract middle slice
        prediction_slice = prediction[:, :, index_z, ...]
        metrics_target_slice = metrics_target[:, :, index_z, ...]

        # Scale output to match ground truth mean intensity
        prediction_mean = torch.mean(prediction_slice) + EPSILON
        metrics_target_mean = torch.mean(metrics_target_slice) + EPSILON
        prediction_normalized = prediction_slice / prediction_mean * metrics_target_mean

        psnr = self.psnr_metric(prediction_normalized, metrics_target_slice)
        ssim = self.ssim_metric(prediction_normalized, metrics_target_slice)

        self.log("val_psnr", psnr)
        self.log("val_ssim", ssim)

    def _normalize_log_image(
        self,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the image tensor for logging to TensorBoard.

        Expects image_tensor to be (C, D, H, W) or (C, H, W).
        Extracts a 2D slice (C, H, W) or (H, W) for logging.
        """
        normalization_method = self.kwargs.get("log_image_normalization", "min-max")
        img_display: torch.Tensor

        # If 4D (C, D, H, W), take the middle depth slice
        if image_tensor.ndim == 4:
            if image_tensor.shape[1] > 0:  # Depth dimension
                img_display = image_tensor[:, image_tensor.shape[1] // 2, :, :]
            else:
                img_display = image_tensor.squeeze(1)
                LOGGER.warning(
                    "Image tensor for logging has unexpected depth shape: %s", image_tensor.shape
                )
        elif image_tensor.ndim == 3:
            img_display = image_tensor
        elif image_tensor.ndim == 2:
            img_display = image_tensor
        else:
            LOGGER.warning(
                "Image tensor for logging has unexpected ndim: %s. Attempting to use as is.",
                image_tensor.shape,
            )
            img_display = image_tensor

        img_display = img_display.detach().cpu().float()

        if img_display.ndim == 3 and img_display.shape[0] != 1 and img_display.shape[0] != 3:
            img_display = img_display[img_display.shape[0] // 2, :, :]

        normalized_img: torch.Tensor
        if torch.allclose(img_display.min(), img_display.max()):
            normalized_img = torch.full_like(
                img_display, 0.0 if normalization_method == "min-max" else 128.0
            )
        else:
            min_val, max_val = img_display.min(), img_display.max()
            match normalization_method:
                case "min-max":
                    normalized_img = (img_display - min_val) / (max_val - min_val + EPSILON) * 255.0
                case "mean-std":
                    mean_val, std_val = img_display.mean(), img_display.std()
                    normalized_img = (img_display - mean_val) / (std_val + EPSILON) * 64.0 + 128.0
                case "mean":
                    mean_val = img_display.mean()
                    normalized_img = img_display / (mean_val + EPSILON) * 128.0
                case _:
                    LOGGER.warning(
                        "Normalization method '%s' not recognized. Using min-max.",
                        normalization_method,
                    )
                    normalized_img = (img_display - min_val) / (max_val - min_val + EPSILON) * 255.0

        # Clamp and convert to uint8
        return torch.clamp(normalized_img, 0, 255).to(torch.uint8)

    def _log_image(
        self,
        image: torch.Tensor,
        name: str,
        batch_idx: int,
        frequency: int = 10,
    ) -> None:
        """Log the image."""
        if batch_idx % frequency == 0:
            img = self._normalize_log_image(image)
            self.logger.experiment.add_image(name, img, self.current_epoch)  # type: ignore

    def _check_conflicts(self) -> None:
        """Check for conflicts in the model configuration."""
        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if (
            self.config.up_mode == "upsample" and self.config.merge_mode == "add"
        ) or self.config.up_mode not in ["upsample", "transpose", "pixelshuffle"]:
            raise InvalidUpModeError(self.config.up_mode)

    def test_step(
        self,
        batch: list[torch.Tensor],  # batch of test data
        batch_idx: int,
    ) -> None:
        """Test step of the model."""
        y_loss_target = batch[0]
        x = batch[1]
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_loss_target)
        self._tb_test_log(loss, y_pred, y_loss_target, batch_idx)

    def _tb_test_log(
        self,
        loss: torch.Tensor,
        prediction: torch.Tensor,
        loss_target: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Log the test step. Can be extended for logging metrics."""
        self.log("test_loss", loss)

    def on_train_end(self) -> None:
        """Save the model at the end of training."""
        self.trainer.save_checkpoint(self.trainer.default_root_dir)

    def reset_lr(self) -> None:
        """Reset the learning rate."""
        self.optimizers().param_groups[0]["lr"] = self.config.optimizer_config["lr"]  # type: ignore

    def freeze_encoder(self) -> None:
        """Freeze encoder layers for transfer learning."""
        for param in self.down_convs.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder layers."""
        for param in self.down_convs.parameters():
            param.requires_grad = True
