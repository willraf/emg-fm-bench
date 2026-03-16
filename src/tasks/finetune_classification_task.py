""" Module classification_task.py

Task for finetuning a foundation model on a classification task

Usage:


Authors:
    Will Raftery

Date:
    10/03/2026
"""

from src.tasks.base_task import BaseTask
from src.models import setup_model
import src.utils.visualisation as vis
from src.utils.metrics import Metrics
from src.models.bio_fm_wrapper import BioFMWrapper 

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging
from pathlib import Path
import torch.distributed as dist
import numpy as np


class FinetuneClassificationTask(BaseTask):
    def __init__(self, cfg, device, distributed=False, is_main=False, save_dir=None):
        super().__init__()

        self.cfg = cfg

        lr = cfg.finetuning.learning_rate
        wd = cfg.finetuning.weight_decay
        num_epochs = cfg.training.epochs
        warmup_epochs = cfg.finetuning.warmup_epochs
        use_amp = cfg.training.get("use_amp", False)

        self.target_field = cfg.finetuning.target_field

        freeze_backbone = self.cfg.finetuning.freeze_backbone
        from_scratch = self.cfg.finetuning.from_scratch
        
        self.device = device
        self.distributed = distributed
        self.is_main = is_main
        self.save_dir = save_dir

        fm = setup_model(cfg.model.fm, device=device)

        if not from_scratch:
            fm_path = cfg.model.fm.checkpoint
            fm_path = Path(fm_path)

            if fm_path.exists():
                logging.info(f"Loading foundation model from checkpoint: {fm_path}")
            else:
                logging.warning(f"Foundation model checkpoint not found: {fm_path}. Training from scratch instead.")
            fm.load_checkpoint(fm_path)
        else:
            logging.info("Training foundation model from scratch. No pretrained weights will be loaded.")

        fm.to(device)

        self.model = BioFMWrapper(cfg.model.wrapper, fm)
        self.model.to(device)

        if freeze_backbone:
            # Freeze everything
            for p in self.model.parameters():
                p.requires_grad = False

            finetune_params = self.model.get_fine_tuning_parameters()

            # Unfreeze parameters to be fine-tuned
            for p in finetune_params:
                p.requires_grad = True

        self.optimiser = self.setup_optimiser(lr, wd)
        self.scheduler = self.setup_scheduler(self.optimiser, warmup_epochs=warmup_epochs, total_epochs=num_epochs)

        # L40/40-series support bfloat16 which is more stable for Transformers
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Scaler is only strictly needed for float16, but harmless for bfloat16
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=(self.use_amp and self.amp_dtype == torch.float16))

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_metrics = Metrics(device, distributed, is_main)
        self.val_metrics = Metrics(device, distributed, is_main)
        self.test_metrics = Metrics(device, distributed, is_main)


    def train_step(self, batch, batch_idx=None, epoch=None):
        emg = batch["emg"].to(self.device)  # (B, C, T)
        targets = batch[self.target_field].long().to(self.device)  # (B,)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            logits = self.model(emg)  
            loss = self.loss_fn(logits, targets)

        self.optimiser.zero_grad(set_to_none=True)

        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()
        else:
            # For BF16 or FP32
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

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            logits = self.model(emg)
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

        logits = self.model(emg)
        loss = self.loss_fn(logits, targets)
        preds = logits.argmax(dim=1)
        
        self.test_metrics.update(preds, targets, loss=loss, num_samples=targets.size(0))
        
        return loss.item()


    def save_checkpoint(self, checkpoint_path, epoch, val_loss):
        # Remove DDP wrapper for saving
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint


    def wrap_distributed(self, local_rank):
        self.model = DDP(self.model, 
                         device_ids=[local_rank], 
                         output_device=local_rank,
                         find_unused_parameters=True  # Allow for some parameters to be frozen or unused
                         )


    def setup_optimiser(self, lr, weight_decay):
        """
        Groups parameters: 
        - Blacklist: Biases, Normalization (LayerNorm, BatchNorm), and Embeddings (WD=0)
        - All others: Receive weight decay.
        """
        decay = set()
        no_decay = set()
        
        # Blacklist logic: anything that shouldn't be squeezed toward zero
        blacklist_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm1d, 
                             torch.nn.BatchNorm2d) #, torch.nn.Embedding)
        
        param_to_module = {}
        for m in self.model.modules():
            for pn, p in m.named_parameters(recurse=False):
                param_to_module[p] = m

        for pn, p in self.model.named_parameters():
            # Skip frozen parameters
            if not p.requires_grad:
                continue

            # Get the parent module of this specific parameter
            container_module = param_to_module[p]

            # 1. Biases always no_decay
            if pn.endswith('bias'):
                no_decay.add(pn)
            
            # 2. Normalization layers (LayerNorm, BatchNorm) always no_decay
            elif isinstance(container_module, blacklist_modules):
                no_decay.add(pn)
            
            # 3. Special tokens (transformer specific)
            elif any(x in pn for x in ['pos_embed', 'cls_token', 'time_embed']):
                no_decay.add(pn)
            
            # 4. Everything else (Linear weights, Conv weights, etc.)
            else:
                decay.add(pn)

        # Sanity Checks
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert len(inter_params) == 0, f"Conflict: {inter_params} in both groups!"
        assert len(param_dict.keys() - union_params) == 0, f"Forgotten: {param_dict.keys() - union_params}"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        logging.info(f"Optimiser groups: {len(decay)} params with WD, {len(no_decay)} params without.")
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
    

    def setup_scheduler(self, optimiser, warmup_epochs, total_epochs):
        """
        Creates a sequential scheduler: 
        1. Linear Warmup (0 -> Base LR)
        2. Cosine Annealing (Base LR -> eta_min)
        """
        # 1. Linear Warmup: Start at a tiny fraction of LR and climb to 1.0x LR
        # We use a multiplier factor: 0.1 means start at 10% of your target LR
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimiser, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        # 2. Cosine Annealing: Decay from target LR down to eta_min
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, 
            T_max=(total_epochs - warmup_epochs),
            eta_min=1e-6
        )
        
        # 3. Combine them
        # milestone is the epoch index where the switch happens
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )
        
        return scheduler