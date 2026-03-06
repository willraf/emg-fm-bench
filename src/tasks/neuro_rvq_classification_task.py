""" Module neuro_rvq_eval_task.py

Finetune neuroRVQ on different datasets

Usage:


Authors:
    Will Raftery

Date:
    23/02/2026
"""


from src.tasks.base_task import BaseTask
from src.models import setup_model
import src.utils.visualisation as vis
from src.utils.metrics import Metrics
from src.models.neuro_rvq_wrapper import NeuroRVQClassifierWrapper 

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging
from pathlib import Path
import torch.distributed as dist
import numpy as np


class NeuroRVQClassificationTask(BaseTask):
    def __init__(self, cfg, device, distributed=False, is_main=False, save_dir=None):


        super().__init__()

        self.cfg = cfg

        fm_path = cfg.model.fm.checkpoint
        lr = cfg.finetuning.learning_rate
        wd = cfg.finetuning.weight_decay
        self.target_field = cfg.finetuning.target_field

        self.patch_len = cfg.model.fm.patch_size

        freeze_backbone = self.cfg.finetuning.freeze_backbone
        
        self.device = device
        self.distributed = distributed
        self.is_main = is_main
        self.save_dir = save_dir

        fm = setup_model(cfg.model.fm, model_name="neuro_rvq_fm", device=device)
        self.load_fm_checkpoint(fm_path, fm)
        fm.to(device)

        if freeze_backbone:
            # Freeze everything
            for p in fm.parameters():
                p.requires_grad = False

            self.model = NeuroRVQClassifierWrapper(cfg, fm)
            self.model.to(device)

            finetune_params = self.model.get_fine_tuning_parameters()

            # Unfreeze parameters to be fine-tuned
            for p in finetune_params:
                p.requires_grad = True
        else:
            self.model = NeuroRVQClassifierWrapper(cfg, fm)
            self.model.to(device)
            finetune_params = self.model.parameters()
                
        self.optimiser = torch.optim.AdamW(
            finetune_params,
            lr=lr,
            weight_decay=wd
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_metrics = Metrics(device, distributed, is_main)
        self.val_metrics = Metrics(device, distributed, is_main)
        self.test_metrics = Metrics(device, distributed, is_main)


    # def resolve_parameters(self):
    #     """Ensure parameters given in the config file are consistent wih each other
    #     window_duration, window_lenght, target_length, n_patches

    #     This is super hacky but will do for now. 
    #     """

    #     window_duration = self.cfg.window_duration
    #     sample_rate = self.cfg.data.sample_rate
    #     patch_length = self.cfg.model.fm.patch_size
    #     target_fs = self.cfg.resample.target_fs

    #     window_length = int(window_duration * sample_rate)
    #     if window_length != self.cfg.data.window_length:
    #         logging.warning(f"Derived window_length {window_length} from window_duration and sample_rate does not match config value {self.cfg.data.window_length}. Using derived value.")
    #         self.cfg.data.window_length = window_length

    #     if target_fs != 1000:
    #         logging.warning(f"Target sample rate {target_fs} is not 1000Hz. Sample rate is expected to be 1000Hz for NeuroRVQ. Ensure resampling is configured correctly.") 

    #     # Check we are padding to the correct length
    #     target_len_after_resampling = int(window_duration * target_fs)
    #     if target_len_after_resampling != self.cfg.pad_length.target_length:
    #         logging.warning(f"Derived target_length after resampling {target_len_after_resampling} does not match pad_length.target_length {self.cfg.pad_length.target_length}. Adjusting pad_length.target_length to match derived value.")
    #         self.cfg.pad_length.target_length = target_len_after_resampling


    def train_step(self, batch, batch_idx=None, epoch=None):
        temporal_ix = batch["temporal_ix"].int().to(self.device)   # B, n_patches  
        spatial_ix = batch["spatial_ix"].int().to(self.device)      # B, n_patches
        emg = batch["emg"].to(self.device)  # (B, C, T)
        targets = batch[self.target_field].long().to(self.device)  # (B,)

        B, C, T = emg.shape

        emg = emg.reshape(B, C, -1, self.patch_len)

        logits = self.model(
            emg,
            temporal_embedding_ix=temporal_ix,
            spatial_embedding_ix=spatial_ix
        )  
        
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
        temporal_ix = batch["temporal_ix"].int().to(self.device)
        spatial_ix = batch["spatial_ix"].int().to(self.device)
        targets = batch[self.target_field].long().to(self.device)  # (B,)
        B, C, T = emg.shape
        emg = emg.reshape(B, C, -1, self.patch_len)
        logits = self.model(
            emg,
            temporal_embedding_ix=temporal_ix,
            spatial_embedding_ix=spatial_ix
        )
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
        
        # ... (same reshaping logic as val_step) ...
        temporal_ix = batch["temporal_ix"].int().to(self.device)
        spatial_ix = batch["spatial_ix"].int().to(self.device)
        B, C, T = emg.shape
        emg = emg.reshape(B, C, -1, self.patch_len)

        logits = self.model(
            emg,
            temporal_embedding_ix=temporal_ix,
            spatial_embedding_ix=spatial_ix
        )
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
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        return checkpoint


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
