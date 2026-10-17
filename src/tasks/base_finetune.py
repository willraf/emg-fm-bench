""" Module base_finetune_task.py

A base class for fine-tuning tasks, providing shared infrastructure and utilities.

Usage:

Date:
    19/06/2026
"""

import contextlib

from src.tasks.base_task import BaseTask
from src.models import setup_model
from src.utils.metrics import RegressionMetrics
from src.models.bio_fm_wrapper import BioFMWrapper 

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging
from pathlib import Path
import time


class BaseFinetuneTask(BaseTask):
    """
    Shared infrastructure for all fine-tuning tasks.

    Subclasses must implement:
        _prepare_targets(batch) -> torch.Tensor
        _compute_loss(preds, targets) -> torch.Tensor
        and instantiate self.train_metrics, self.val_metrics, self.test_metrics
    """

    def __init__(self, cfg, device, distributed=False, is_main=False):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.distributed = distributed
        self.is_main = is_main

        self.num_accumulation_steps = cfg.training.get("num_accumulation_steps", 1)
        self.use_amp = cfg.task.get("use_amp", False)

        num_epochs = cfg.training.epochs
        lr = cfg.task.learning_rate
        wd = cfg.task.weight_decay
        warmup_epochs = cfg.task.warmup_epochs
        freeze_backbone = cfg.task.freeze_backbone
        from_scratch = cfg.task.from_scratch
        
        # ----------------------------------------
        #           Foundation Model Setup
        # ----------------------------------------

        fm = setup_model(cfg.model.fm, device=device)

        if not from_scratch:
            fm_path = cfg.model.fm.checkpoint
            fm_path = Path(fm_path)

            if fm_path.exists():
                logging.info(f"Loading foundation model from checkpoint: {fm_path}")
                fm.load_checkpoint(fm_path)
            else:
                # logging.warning(f"Foundation model checkpoint not found: {fm_path}. Training from scratch instead.")
                raise FileNotFoundError(f"Foundation model checkpoint not found: {fm_path}. Please provide a valid checkpoint or set 'from_scratch' to True.")
        else:
            logging.info("Training foundation model from scratch. No pretrained weights will be loaded.")

        fm.to(device)

        # Print number of parameters in the backbone for reference
        num_fm_params = sum(p.numel() for p in fm.parameters())
        logging.info(f"Foundation model parameters: {num_fm_params:,}")

        # ----------------------------------------
        #      Wrapper Model Setup & Freezing
        # ----------------------------------------

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
        

        # Print total number of parameters and trainable parameters for reference
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total model parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")

        # ----------------------------------------
        #      Optimiser, Scheduler & AMP
        # ----------------------------------------
        self.optimiser = self.setup_optimiser(lr, wd)
        self.scheduler = self.setup_scheduler(self.optimiser, warmup_epochs=warmup_epochs, total_epochs=num_epochs)

        # L40/40-series support bfloat16 which is more stable for Transformers
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Scaler is only strictly needed for float16, but harmless for bfloat16
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=(self.use_amp and self.amp_dtype == torch.float16))


    def _prepare_inputs(self, batch) -> torch.Tensor:
        """Defaults to extracting the EMG signal from the batch. Subclasses can override if needed."""
        return batch["emg"].to(self.device)  # (B, C, T)

    def _prepare_targets(self, batch) -> torch.Tensor:
        """Extract and prepare targets. Must be implemented by subclass."""
        raise NotImplementedError

    def _compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute task loss. Must be implemented by subclass."""
        raise NotImplementedError


    def train_step(self, batch, is_update_step=True, batch_idx=None, epoch=None):
        inputs = self._prepare_inputs(batch)  
        targets = self._prepare_targets(batch)  

        # Check the model is in training mode (defensive)
        if not self.model.training:
            logging.warning("Model is not in training mode during train_step. This may cause issues with certain layers like dropout or batchnorm. Make sure to call model.train() before training.")
        # Check pytorch gradients are enabled (defensive)
        if not torch.is_grad_enabled():
            logging.warning("PyTorch gradients are not enabled during train_step. This will prevent the model from learning. Make sure to call torch.set_grad_enabled(True) before training.")

        # This basically means "don't sync gradients across processes during the backward pass if this isn't an update step"
        # It allows gradient accumulation with DDP training
        maybe_no_sync = (
            self.model.no_sync()
            if (self.distributed and not is_update_step)
            else contextlib.nullcontext()
        )

        with maybe_no_sync:
            # This stuff is for amp scaling for memory efficiency
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                preds = self.model(inputs) 
                unscaled_loss = self._compute_loss(preds, targets)
                # We need to do this as default reduction is 'mean' so we need to scale down loss
                loss = unscaled_loss / self.num_accumulation_steps 

            # Always perform (scaled) backward pass, even if not updating weights yet.
            # Prevents needing to store multiple forward passes in memory
            self.scaler.scale(loss).backward()

        # Only update weights if its an update step
        if is_update_step:
            self.scaler.unscale_(self.optimiser)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.)   
            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.optimiser.zero_grad(set_to_none=True)

        with torch.no_grad():
            self.train_metrics.update(preds, targets, loss=unscaled_loss, num_samples=targets.size(0))

        return loss.item()     


    @torch.no_grad()
    def _inference_step(self, batch, metrics=None):
        inputs  = self._prepare_inputs(batch)
        targets = self._prepare_targets(batch)

        if self.model.training:
            logging.warning("Model in training mode during inference step.")
        if torch.is_grad_enabled():
            logging.warning("Gradients enabled during inference step.")

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
            preds = self.model(inputs)
            loss  = self._compute_loss(preds, targets)

        if metrics is not None:
            metrics.update(preds, targets, loss=loss, num_samples=targets.size(0))

        return loss.item()

    def val_step(self, batch):
        return self._inference_step(batch, self.val_metrics)

    def eval_step(self, batch):
        return self._inference_step(batch, self.test_metrics)


    def save_checkpoint(self, checkpoint_path, epoch, val_loss, train_time):
        # Remove DDP wrapper for saving
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_time': train_time
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")


    def load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
    
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Strip DDP prefix if present (defensive)
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logging.info(f"Checkpoint loaded: {checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')}, val_loss {checkpoint.get('val_loss', 'N/A'):.4f})")
        return checkpoint

    def wrap_distributed(self, local_rank):
        if self.cfg.experiment.get("ddp_debug_mode", False):
            logging.warning("This is set to debug mode with DDP wrapping on a single GPU. Make sure to set up distributed training properly for multi-GPU runs!")
            local_rank = 0

        self.model = DDP(self.model, 
                         device_ids=[local_rank], 
                         output_device=local_rank,
                        #  find_unused_parameters=True  # Allow for some parameters to be frozen or unused
                        )
        
    
    def modules(self):
        return [self.model]


    def setup_optimiser(self, lr, weight_decay):
        """
        Groups parameters: 
        - Blacklist: Biases, Normalization (LayerNorm, BatchNorm), and Embeddings (WD=0)
        - All others: Receive weight decay.

        Must be called before wrap_distributed() — operates on the unwrapped model.
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
    
    def measure_inference_latency(self, batch, n_warmup=50, n_runs=200):
        """
        Measure per-sample inference latency with proper GPU warmup.
        Returns mean and std latency in milliseconds.
        """
        self.model.eval()
        emg = batch["emg"].to(self.device)  # (B, C, T)
        sample_input = emg[:1]  # Take a single sample for latency measurement

        # Warmup — not measured
        with torch.inference_mode():
            for _ in range(n_warmup):
                _ = self.model(sample_input)

        torch.cuda.synchronize()

        # Timed runs
        latencies = []
        with torch.inference_mode():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = self.model(sample_input)
                torch.cuda.synchronize()  # Wait for GPU to finish
                latencies.append((time.perf_counter() - start) * 1000)  # ms

        mean_ms = sum(latencies) / len(latencies)
        std_ms  = (sum((x - mean_ms) ** 2 for x in latencies) / len(latencies)) ** 0.5

        logging.info(f"Inference latency: {mean_ms:.2f} ± {std_ms:.2f} ms/batch "
                    f"({mean_ms / sample_input.shape[0]:.2f} ms/sample)")
        return mean_ms, std_ms


    def count_active_parameters(self, sample_batch):
        """
        Count parameters that are actually used in the forward pass
        by registering hooks on all parameters.
        """
        used_params = set()

        hooks = []
        for name, param in self.model.named_parameters():
            # Closure to capture name
            def make_hook(n):
                def hook(grad):
                    used_params.add(n)
                return hook
            hooks.append(param.register_hook(make_hook(name)))

        # Run a forward + backward pass to trigger hooks
        self.model.train()
        with torch.enable_grad():
            emg = sample_batch["emg"].to(self.device)
            output = self.model(emg)
            if isinstance(output, torch.Tensor):
                output.sum().backward()
            elif isinstance(output, (tuple, list)):
                output[0].sum().backward()

        # Remove hooks
        for h in hooks:
            h.remove()

        self.model.zero_grad()

        active = {n: p for n, p in self.model.named_parameters() if n in used_params}
        inactive = {n: p for n, p in self.model.named_parameters() if n not in used_params}

        active_count   = sum(p.numel() for p in active.values())
        inactive_count = sum(p.numel() for p in inactive.values())

        return active_count, inactive_count, active, inactive
