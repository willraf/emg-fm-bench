""" Module finetune_regression_task.py

Task for finetuning a foundation model on a regression task

Usage:


Authors:
    Will Raftery

Date:
    18/06/2026
"""
""" finetune_regression_task.py """

from src.tasks.base_finetune_task import BaseFinetuneTask
from src.utils.metrics import RegressionMetrics
import torch
import logging
import torch.nn.functional as F


class VelocityRegressionTask(BaseFinetuneTask):

    def __init__(self, cfg, device, distributed=False, is_main=False):
        super().__init__(cfg, device, distributed, is_main)
        
        # Expected to  be jntang
        self.target_field = cfg.task.target_field

        if self.target_field != "jntang":
            logging.warning(
                f"Expected target_field to be 'jntang', but got '{self.target_field}'. "
                f"Ensure that the dataset returns joint angle labels under the key '{self.target_field}'."
            )

        self.loss_fn = torch.nn.MSELoss()

        self.train_metrics = RegressionMetrics(device, distributed, is_main)
        self.val_metrics   = RegressionMetrics(device, distributed, is_main)
        self.test_metrics  = RegressionMetrics(device, distributed, is_main)

    # def _prepare_targets(self, batch) -> torch.Tensor:
    #     jntang = batch[self.target_field].to(self.device)  # (B, T, J)

    #     assert jntang.ndim == 3, (
    #         f"Expected (B, T, J), got {jntang.shape}. "
    #         f"Check '{self.target_field}' is returned as (B, T, J) by the dataset."
    #     )

    #     velocities = jntang[:, 1:, :] - jntang[:, :-1, :]  # (B, T-1, J)

    #     emg = batch["emg"].to(self.device)              # (B, C, T_emg)
    #     T_emg = emg.shape[2]
    #     patch_len = self.cfg.model.wrapper.patch_size
    #     # Number of patchs in temporal dimension
    #     N = T_emg // patch_len

    #     T_vel = velocities.shape[1]

    #     if T_vel < N:
    #         velocities = self._resample_labels(velocities, target_length=N)  # (B, N, J)
    #     elif T_vel > N:
    #         velocities = velocities[:, :N, :]  # (B, N, J)

    def _prepare_targets(self, batch) -> torch.Tensor:
        """
        Get a single velocity vector per joint over the whole window
        """
        jntang = batch[self.target_field].to(self.device)  # (B, T, J)

        assert jntang.ndim == 3, (
            f"Expected (B, T, J), got {jntang.shape}. "
            f"Check '{self.target_field}' is returned as (B, T, J) by the dataset."
        )

        velocities = jntang[:, 0, :] - jntang[:, -1, :]  # (B, J)
        return velocities

    
    def _resample_labels(self, labels: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Resample joint angle labels to match resampled EMG length.
        Uses linear interpolation — appropriate for smooth kinematic signals.

        Args:
            labels:        (T, J) or (B, T, J)
            target_length: target number of time samples

        Returns:
            resampled labels at target_length
        """
        if labels.ndim == 2:
            # (T, J) -> (1, J, T) for F.interpolate
            x = labels.T.unsqueeze(0)           # (1, J, T)
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
            return x.squeeze(0).T               # (target_length, J)

        elif labels.ndim == 3:
            # (B, T, J) -> (B, J, T) for F.interpolate
            x = labels.permute(0, 2, 1)         # (B, J, T)
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
            return x.permute(0, 2, 1)           # (B, target_length, J)

        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {labels.ndim}D")

    def _compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)
    
