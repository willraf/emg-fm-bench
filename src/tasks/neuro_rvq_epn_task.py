from src.tasks.base_task import BaseTask
from src.models import setup_model
import src.utils.visualisation as vis
from src.models.neuro_rvq_wrapper import NeuroRVQClassifierWrapper 

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging
from pathlib import Path



import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy import signal



class NeuroRVQEPNTask(BaseTask):
    def __init__(self, cfg, device, distributed=False, is_main=False, save_dir=None):


        super().__init__()

        fm_path = cfg.model.fm.checkpoint
        lr = cfg.finetuning.learning_rate
        wd = cfg.finetuning.weight_decay
        self.target_field = cfg.finetuning.target_field

        self.patch_len = cfg.model.fm.patch_size
        self.n_patches = cfg.model.fm.n_patches

        self.cfg = cfg
        self.device = device
        self.distributed = distributed
        self.is_main = is_main
        self.save_dir = save_dir


        fm = setup_model(cfg.model.fm, model_name="neuro_rvq_fm", device=device)
        self.load_from_checkpoint(fm_path, fm)
        fm.to(device)

        # Freeze everything
        for p in fm.parameters():
            p.requires_grad = False

        self.model = NeuroRVQClassifierWrapper(cfg, fm)
        self.model.to(device)

        head_params = self.model.get_fine_tuning_parameters()
                
        self.optimiser = torch.optim.AdamW(
            head_params,
            lr=lr,
            weight_decay=wd
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()



    def train_step(self, batch, batch_idx=None, epoch=None):
        temporal_ix = batch["temporal_ix"].int().to(self.device)   # B, n_patches  
        spatial_ix = batch["spatial_ix"].int().to(self.device)      # B, n_patches
        emg = batch["emg"].to(self.device)  # (B, C, T)
        targets = batch[self.target_field].to(self.device)  # (B,)

        B, C, T = emg.shape
        time_patches = self.n_patches // C

        emg = emg.reshape(B, C, time_patches, self.patch_len)

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
            acc = (preds == targets).float().mean()

        return {
            "loss": loss.item(),
            "acc": acc.item()
        }        


    @torch.no_grad()
    def val_step(self, batch, batch_idx=None, epoch=None):

        emg = batch["emg"].to(self.device)  # (B, C, T)
        temporal_ix = batch["temporal_ix"].int().to(self.device)
        spatial_ix = batch["spatial_ix"].int().to(self.device)
        targets = batch[self.target_field].to(self.device)  # (B,)
        B, C, T = emg.shape
        time_patches = self.n_patches // C
        emg = emg.reshape(B, C, time_patches, self.patch_len)
        logits = self.model(
            emg,
            temporal_embedding_ix=temporal_ix,
            spatial_embedding_ix=spatial_ix
        )
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean()

        return {
            "loss": loss.item(),
            "acc": acc.item()
        }


    def eval_step(self, model, batch, device):
        return 


    def load_from_checkpoint(self, checkpoint_path, model):
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
        self.tokeniser.load_state_dict(checkpoint)
        return checkpoint


    def wrap_distributed(self, local_rank):
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
