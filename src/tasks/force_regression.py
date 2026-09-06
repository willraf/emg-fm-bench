""" Module finetune_regression_task.py

Task for finetuning a foundation model on a regression task

Usage:


Authors:
    Will Raftery

Date:
    18/06/2026
"""
""" finetune_regression_task.py """

from src.tasks.base_finetune import BaseFinetuneTask
from src.utils.metrics import RegressionMetrics
import torch
import logging
import torch.nn.functional as F


class ForceRegressionTask(BaseFinetuneTask):

    def __init__(self, cfg, device, distributed=False, is_main=False):
        super().__init__(cfg, device, distributed, is_main)

        self.target_field = cfg.task.target_field
        self.loss_fn = torch.nn.MSELoss()

        self.train_metrics = RegressionMetrics(device, distributed, is_main)
        self.val_metrics   = RegressionMetrics(device, distributed, is_main)
        self.test_metrics  = RegressionMetrics(device, distributed, is_main)

    
    # def _prepare_targets(self, batch) -> torch.Tensor:
    #     """
    #     Per-patch prediction of joint angles. Averages joint angles over each patch (in time) to produce a single label per patch.
    #     """
    #     raw = batch[self.target_field].to(self.device)  # (B, T, J)

    #     assert raw.ndim == 3, (
    #         f"Expected (B, T, J), got {raw.shape}. "
    #         f"Check '{self.target_field}' is returned as (B, T, J) by the dataset."
    #     )
    
    #     # Resample to match EMG length if necessary
    #     emg = batch["emg"].to(self.device)              # (B, C, T_emg)
    #     T_emg = emg.shape[2]
    #     T_label = raw.shape[1]

    #     # Resample timeseries labels to match EMG length if necessary
    #     if T_label != T_emg:
    #         raw = self._resample_labels(raw, target_length=T_emg)  # (B, T_emg, J)

    #     B, T, J = raw.shape
    #     patch_len = self.cfg.model.wrapper.patch_size
    #     N = T // patch_len

    #     assert N > 0, (
    #         f"patch_len ({patch_len}) > sequence length ({T}). "
    #         f"Check patch_size config."
    #     )

    #     raw = raw[:, :N * patch_len, :]
    #     # raw = raw.mean(dim=1).reshape(B, N, J)  # (B, J)
    #     raw = raw.reshape(B, N, patch_len, J).mean(dim=2)  # (B, N, J)
    #     return raw


    # def _resample_labels(self, labels: torch.Tensor, target_length: int) -> torch.Tensor:
    #     """
    #     Resample joint angle labels to match resampled EMG length.
    #     Uses linear interpolation — appropriate for smooth kinematic signals.

    #     Args:
    #         labels:        (T, J) or (B, T, J)
    #         target_length: target number of time samples

    #     Returns:
    #         resampled labels at target_length
    #     """
    #     if labels.ndim == 2:
    #         # (T, J) -> (1, J, T) for F.interpolate
    #         x = labels.T.unsqueeze(0)           # (1, J, T)
    #         x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
    #         return x.squeeze(0).T               # (target_length, J)

    #     elif labels.ndim == 3:
    #         # (B, T, J) -> (B, J, T) for F.interpolate
    #         x = labels.permute(0, 2, 1)         # (B, J, T)
    #         x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
    #         return x.permute(0, 2, 1)           # (B, target_length, J)


    def _prepare_targets(self, batch) -> torch.Tensor:
        """
        Per-window prediction of joint angles. Takes the final frame of the window as the target label for the whole window.
        """
        raw = batch[self.target_field].to(self.device)  # (B, T, J)
        return raw.mean(dim=1)      # Just return the mean force value


    def _compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)
    
