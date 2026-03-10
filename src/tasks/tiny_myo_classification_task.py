""" Module tiny_myo_classification_task.py

***Add description here***

Usage:


Authors:
    Will Raftery

Date:
    10/03/2026
"""



def main():
    pass

if __name__ == '__main__':
    main()


from src.tasks.base_task import BaseTask
from src.models import setup_model
import src.utils.visualisation as vis
from src.utils.metrics import Metrics
from src.models.tiny_myo import TinyMyo

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging
from pathlib import Path
import torch.distributed as dist
import numpy as np

from safetensors.torch import load_file


class TinyMyoClassificationTask(BaseTask):
    def __init__(self, cfg, device, distributed=False, is_main=False, save_dir=None):

        super().__init__()

        self.cfg = cfg

        fm_path = cfg.model.fm.checkpoint
        lr = cfg.finetuning.learning_rate
        wd = cfg.finetuning.weight_decay
        self.target_field = cfg.finetuning.target_field

        freeze_backbone = self.cfg.finetuning.freeze_backbone
        
        self.device = device
        self.distributed = distributed
        self.is_main = is_main
        self.save_dir = save_dir

        self.model = TinyMyo(
            img_size=cfg.data.window_length,
            in_chans=cfg.data.num_channels,
            num_classes=cfg.finetuning.n_classes,
        )

        self.load_fm_checkpoint(fm_path, self.model)
        self.model.to(device)

        if freeze_backbone:
            # Freeze everything
            for p in self.model.parameters():
                p.requires_grad = False

            finetune_params = list(self.model.model_head.parameters())

            # Unfreeze parameters to be fine-tuned
            for p in finetune_params:
                p.requires_grad = True
        else:
            finetune_params = list(self.model.parameters())
                
        self.optimiser = torch.optim.AdamW(
            finetune_params,
            lr=lr,
            weight_decay=wd
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_metrics = Metrics(device, distributed, is_main)
        self.val_metrics = Metrics(device, distributed, is_main)
        self.test_metrics = Metrics(device, distributed, is_main)


    def train_step(self, batch, batch_idx=None, epoch=None):
        emg = batch["emg"].to(self.device)  # (B, C, T)
        targets = batch[self.target_field].long().to(self.device)  # (B,)

        logits, _ = self.model(emg)  
        
        loss = self.loss_fn(logits, targets)
        
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.optimiser.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)

        self.train_metrics.update(preds, targets, loss=loss, num_samples=targets.size(0))

        return loss.item()     


    @torch.no_grad()
    def val_step(self, batch, batch_idx=None, epoch=None):

        emg = batch["emg"].to(self.device)  # (B, C, T)
        targets = batch[self.target_field].long().to(self.device)  # (B,)
        logits, _ = self.model(emg)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        
        self.val_metrics.update(preds, targets, loss=loss, num_samples=targets.size(0))

        return loss.item()     


    @torch.no_grad()
    def eval_step(self, batch):
        """
        Standardized evaluation step. Returns raw outputs and targets.
        """
        emg = batch["emg"].to(self.device)
        targets = batch[self.target_field].to(self.device)

        logits, _ = self.model(emg)
        preds = logits.argmax(dim=1)

        loss = self.loss_fn(logits, targets)
        
        self.test_metrics.update(preds, targets, loss=loss, num_samples=targets.size(0))
        
        return loss.item()

    def load_fm_checkpoint(self, checkpoint_path, model):
        """
        Load the pretrained foundation model weights.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
        
        state = load_file(path)

        # Keep only the backbone params and strip the leading "model."
        adapted_state = {}
        for k, v in state.items():
            if k.startswith("model."):
                new_k = k[len("model."):]  # drop the "model." prefix
                adapted_state[new_k] = v

        model.load_state_dict(adapted_state, strict=False)  # strict=False allows loading only the matching keys, ignoring any missing or unexpected keys
        return None


    def save_checkpoint(self, checkpoint_path, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")


    def load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


    def wrap_distributed(self, local_rank):
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
