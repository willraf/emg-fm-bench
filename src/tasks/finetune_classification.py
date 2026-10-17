""" Module classification_task.py

Task for finetuning a foundation model on a classification task

Usage:


Date:
    10/03/2026
"""

from src.tasks.base_finetune import BaseFinetuneTask
from src.utils.metrics import ClassificationMetrics
import torch


class FinetuneClassificationTask(BaseFinetuneTask):

    def __init__(self, cfg, device, distributed=False, is_main=False):
        super().__init__(cfg, device, distributed, is_main)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_metrics = ClassificationMetrics(device, distributed, is_main)
        self.val_metrics   = ClassificationMetrics(device, distributed, is_main)
        self.test_metrics  = ClassificationMetrics(device, distributed, is_main)

        self.target_field = cfg.task.target_field

    def _prepare_targets(self, batch) -> torch.Tensor:
        return batch[self.target_field].to(self.device)  # (B,)

    def _compute_loss(self, preds, targets):
        return self.loss_fn(preds, targets)