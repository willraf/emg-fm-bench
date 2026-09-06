""" Module flex_pretraining_task.py

Task for pretraining the FlexEMG model using masked patch prediction
with an EMA target encoder (I-JEPA style).

Pretraining objective:
    - Randomly mask N% of patches in the online encoder input
    - Online encoder sees masked input, produces embeddings
    - Target encoder sees FULL unmasked input, produces target embeddings
    - Loss computed between online predictions and target embeddings
      for masked patches only

Authors:
    Will Raftery
Date:
    09/07/2026
"""

import contextlib
import copy
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.tasks.base_task import BaseTask
from src.models import setup_model
from src.utils.metrics import RegressionMetrics

# --------------------------------------------------------------------------- #
# EMA utilities
# --------------------------------------------------------------------------- #

def build_ema_encoder(online_encoder: nn.Module) -> nn.Module:
    """
    Create a target encoder as a deep copy of the online encoder.
    All parameters are frozen — the target encoder is updated via EMA only.

    Args:
        online_encoder: the online encoder module

    Returns:
        target encoder with identical weights, all grads disabled
    """
    target = copy.deepcopy(online_encoder)
    for p in target.parameters():
        p.requires_grad = False
    target.eval()
    return target


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, tau: float):
    """
    Update target encoder weights via exponential moving average.

        theta_target = tau * theta_target + (1 - tau) * theta_online

    Args:
        online: online encoder (being trained)
        target: target encoder (EMA copy)
        tau:    momentum coefficient, typically 0.996-0.9999
    """
    # Unwrap DDP if necessary
    online_params = (
        online.module.parameters()
        if isinstance(online, DDP)
        else online.parameters()
    )
    target_params = target.parameters()

    for p_online, p_target in zip(online_params, target_params):
        p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


def cosine_ema_schedule(
    base_tau: float,
    final_tau: float,
    current_step: int,
    total_steps: int,
) -> float:
    """
    Cosine schedule for EMA momentum — starts at base_tau and increases
    to final_tau over training. Higher tau = slower target update = more
    stable targets early in training.

    Args:
        base_tau:     starting momentum (e.g. 0.996)
        final_tau:    ending momentum (e.g. 0.9999)
        current_step: current training step
        total_steps:  total number of training steps

    Returns:
        tau for current step
    """
    import math
    return final_tau - (final_tau - base_tau) * (
        math.cos(math.pi * current_step / total_steps) + 1
    ) / 2


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #

def random_mask(
    B: int, C: int, N: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a random binary mask over patches.

    Args:
        B:          batch size
        C:          number of channels
        N:          number of temporal patches
        mask_ratio: fraction of patches to mask
        device:     torch device

    Returns:
        mask: (B, C, N) bool tensor — True = masked
    """
    n_mask = max(1, int(N * mask_ratio))
    noise  = torch.rand(B, C, N, device=device)
    ids    = noise.argsort(dim=-1)
    return ids < n_mask   # (B, C, N)


# --------------------------------------------------------------------------- #
# Pretraining Task
# --------------------------------------------------------------------------- #

class FlexPretrainingTask(BaseTask):
    """
    Pretraining task for FlexEMG using masked patch prediction with
    an EMA target encoder.

    Online encoder:  sees masked input → predicts latent embeddings
    Target encoder:  sees full input   → produces stable target embeddings
    Loss:            smooth L1 between online predictions and target
                     embeddings, computed on masked patches only

    Args:
        cfg:         Hydra config
        device:      torch device
        distributed: whether DDP is active
        is_main:     whether this is the main process
    """

    def __init__(self, cfg, device, distributed=False, is_main=False):
        super().__init__(cfg)

        self.cfg         = cfg
        self.device      = device
        self.distributed = distributed
        self.is_main     = is_main

        self.num_accumulation_steps = cfg.training.get("num_accumulation_steps", 1)
        self.use_amp     = cfg.task.get("use_amp", False)
        self.mask_ratio  = cfg.task.get("mask_ratio", 0.5)

        # EMA schedule parameters
        self.ema_tau_base  = cfg.task.get("ema_tau_base",  0.996)
        self.ema_tau_final = cfg.task.get("ema_tau_final", 0.9999)
        self.global_step   = 0
        self.total_steps   = None   # set in train() once we know steps per epoch

        num_epochs    = cfg.training.epochs
        lr            = cfg.task.learning_rate
        wd            = cfg.task.weight_decay
        warmup_epochs = cfg.task.warmup_epochs

        # ------------------------------------------------------------------ #
        # Online encoder
        # ------------------------------------------------------------------ #
        self.online_encoder = setup_model(cfg.model.encoder, device=device)

        n_params = sum(p.numel() for p in self.online_encoder.parameters())
        logging.info(f"FlexEMG online encoder parameters: {n_params:,}")

        # ------------------------------------------------------------------ #
        # Target encoder — EMA copy, never directly optimised
        # ------------------------------------------------------------------ #
        self.target_encoder = build_ema_encoder(self.online_encoder)
        self.target_encoder.to(device)

        n_target = sum(p.numel() for p in self.target_encoder.parameters())
        logging.info(f"FlexEMG target encoder parameters: {n_target:,} (EMA, no grad)")

        # ------------------------------------------------------------------ #
        # Predictor head
        # Projects online encoder output to target encoder embedding space.
        # Keeping this separate from the encoder prevents the online encoder
        # from collapsing to a trivial solution.
        # ------------------------------------------------------------------ #
        embed_dim = cfg.model.encoder.embed_dim
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        ).to(device)

        # ------------------------------------------------------------------ #
        # Mask token — replaces masked patch embeddings in online encoder
        # ------------------------------------------------------------------ #
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, embed_dim, device=device)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ------------------------------------------------------------------ #
        # Optimiser, scheduler, AMP
        # Only online encoder + predictor + mask_token are optimised
        # ------------------------------------------------------------------ #
        self.optimiser = self.setup_optimiser(lr, wd)
        self.scheduler = self.setup_scheduler(
            self.optimiser, warmup_epochs=warmup_epochs, total_epochs=num_epochs
        )

        self.amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )

        logging.info(
            f"FlexPretrainingTask initialised — "
            f"mask_ratio={self.mask_ratio}, "
            f"ema_tau=[{self.ema_tau_base}, {self.ema_tau_final}]"
        )

        self.train_metrics = RegressionMetrics(device=device, distributed=distributed, is_main=is_main)
        self.val_metrics   = RegressionMetrics(device=device, distributed=distributed, is_main=is_main)
        self.test_metrics  = RegressionMetrics(device=device, distributed=distributed, is_main=is_main)

    # ---------------------------------------------------------------------- #
    # Forward helpers
    # ---------------------------------------------------------------------- #

    def _forward_online(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through online encoder with masking.

        Replaces masked patch embeddings with the learned mask token before
        passing through the transformer, so the model must predict masked
        content from context.

        Args:
            x:    (B, C, T) raw EMG
            mask: (B, C, N) bool — True = masked

        Returns:
            (B, C, N, D) online embeddings
        """
        encoder = (
            self.online_encoder.module
            if isinstance(self.online_encoder, DDP)
            else self.online_encoder
        )

        B, C, T = x.shape
        P = encoder.patch_size
        N = T // P

        # Patch embed + positional embeddings
        tokens = encoder.patch_embed(x)              # (B, C, N, D)
        tokens = tokens + encoder.temporal_pe(N)
        tokens = tokens + encoder.spatial_pe(C)

        # Replace masked positions with mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(
            mask_expanded,
            self.mask_token.expand_as(tokens),
            tokens,
        )

        # Transformer
        B, C, N, D = tokens.shape
        tokens = tokens.reshape(B, C * N, D)
        for block in encoder.blocks:
            tokens = block(tokens)
        tokens = encoder.norm(tokens)
        tokens = tokens.reshape(B, C, N, D)

        return tokens

    @torch.no_grad()
    def _forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through target encoder — full unmasked input.
        No gradients — target encoder is updated via EMA only.

        Args:
            x: (B, C, T) raw EMG

        Returns:
            (B, C, N, D) target embeddings
        """
        return self.target_encoder(x)

    def _compute_loss(
        self,
        online_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Smooth L1 loss between online predictions and target embeddings,
        computed only on masked patches.

        Args:
            online_tokens: (B, C, N, D) — online encoder output
            target_tokens: (B, C, N, D) — target encoder output (no grad)
            mask:          (B, C, N)    — True = masked

        Returns:
            scalar loss
        """
        # Project online embeddings to prediction space
        preds   = self.predictor(online_tokens)   # (B, C, N, D)

        # Normalise targets — prevents scale collapse
        # (predicting normalised targets is more stable than raw embeddings)
        targets = F.layer_norm(
            target_tokens,
            normalized_shape=(target_tokens.shape[-1],)
        )

        # Loss only on masked patches
        return F.smooth_l1_loss(preds[mask], targets[mask].detach())

    # ---------------------------------------------------------------------- #
    # Train / inference steps
    # ---------------------------------------------------------------------- #

    def train_step(self, batch, is_update_step=True, batch_idx=None, epoch=None):
        emg = batch["emg"].to(self.device)   # (B, C, T)
        B, C, T = emg.shape

        # Unwrap encoder to get patch size attribute
        encoder = (
            self.online_encoder.module
            if isinstance(self.online_encoder, DDP)
            else self.online_encoder
        )
        N = T // encoder.patch_size

        if not self.online_encoder.training:
            logging.warning("Online encoder not in training mode during train_step.")
        if not torch.is_grad_enabled():
            logging.warning("Gradients disabled during train_step.")

        # Generate mask
        mask = random_mask(B, C, N, self.mask_ratio, self.device)

        # This basically means "don't sync gradients across processes during the backward pass if this isn't an update step"
        # It allows gradient accumulation with DDP training
        maybe_no_sync = (
            self.online_encoder.no_sync()
            if (self.distributed and not is_update_step)
            else contextlib.nullcontext()
        )

        with maybe_no_sync:
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
                dtype=self.amp_dtype,
            ):
                # Online forward (masked)
                online_tokens = self._forward_online(emg, mask)   # (B, C, N, D)

                # Target forward (unmasked, no grad)
                target_tokens = self._forward_target(emg)         # (B, C, N, D)

                unscaled_loss = self._compute_loss(
                    online_tokens, target_tokens, mask
                )
                loss = unscaled_loss / self.num_accumulation_steps

            self.scaler.scale(loss).backward()

        if is_update_step:
            self.scaler.unscale_(self.optimiser)
            trainable = [
                p for p in list(self.online_encoder.parameters())
                       + list(self.predictor.parameters())
                       + [self.mask_token]
                if p.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.optimiser.zero_grad(set_to_none=True)

            # EMA update — only on update steps
            tau = cosine_ema_schedule(
                self.ema_tau_base,
                self.ema_tau_final,
                self.global_step,
                self.total_steps or 1,
            )
            update_ema(self.online_encoder, self.target_encoder, tau)
            self.global_step += 1

        return unscaled_loss.item()

    @torch.no_grad()
    def _inference_step(self, batch, metrics=None):
        emg = batch["emg"].to(self.device)
        B, C, T = emg.shape

        encoder = (
            self.online_encoder.module
            if isinstance(self.online_encoder, DDP)
            else self.online_encoder
        )
        N = T // encoder.patch_size

        mask = random_mask(B, C, N, self.mask_ratio, self.device)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
            dtype=self.amp_dtype,
        ):
            online_tokens = self._forward_online(emg, mask)
            target_tokens = self._forward_target(emg)
            loss = self._compute_loss(online_tokens, target_tokens, mask)

        return loss.item()

    def val_step(self, batch):
        return self._inference_step(batch)

    def eval_step(self, batch):
        return self._inference_step(batch)

    # ---------------------------------------------------------------------- #
    # Infrastructure
    # ---------------------------------------------------------------------- #

    def set_total_steps(self, steps_per_epoch: int, num_epochs: int):
        """Call this from the trainer once DataLoader length is known."""
        self.total_steps = steps_per_epoch * num_epochs
        logging.info(f"EMA schedule: {self.total_steps} total steps, "
                     f"tau {self.ema_tau_base} → {self.ema_tau_final}")

    def save_checkpoint(self, checkpoint_path, epoch, val_loss, train_time):
        torch.save({
            "epoch":                    epoch,
            "val_loss":                 val_loss,
            "train_time":               train_time,
            "online_encoder_state":     self._unwrap(self.online_encoder).state_dict(),
            "target_encoder_state":     self.target_encoder.state_dict(),
            "predictor_state":          self.predictor.state_dict(),
            "mask_token":               self.mask_token.data,
            "optimizer_state_dict":     self.optimiser.state_dict(),
            "scheduler_state_dict":     self.scheduler.state_dict(),
            "global_step":              self.global_step,
        }, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
        ckpt = torch.load(path, map_location=self.device)
        self._unwrap(self.online_encoder).load_state_dict(ckpt["online_encoder_state"])
        self.target_encoder.load_state_dict(ckpt["target_encoder_state"])
        self.predictor.load_state_dict(ckpt["predictor_state"])
        self.mask_token.data = ckpt["mask_token"]
        self.optimiser.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        logging.info(
            f"Checkpoint loaded: {path} "
            f"(epoch {ckpt.get('epoch', 'N/A')}, "
            f"val_loss {ckpt.get('val_loss', 'N/A'):.4f})"
        )
        return ckpt

    def wrap_distributed(self, local_rank):
        if self.cfg.experiment.get("ddp_debug_mode", False):
            local_rank = 0
        self.online_encoder = DDP(
            self.online_encoder,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        # Target encoder is NOT wrapped in DDP — it has no gradients

    def modules(self):
        return [self.online_encoder]

    def _unwrap(self, model):
        return model.module if isinstance(model, DDP) else model

    def setup_optimiser(self, lr, weight_decay):
        """
        AdamW with weight decay on weights only.
        Optimises: online encoder + predictor + mask token.
        Target encoder is excluded — updated via EMA.
        """
        decay, no_decay = set(), set()
        blacklist = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)

        # Collect all trainable parameters
        named_params = (
            list(self._unwrap(self.online_encoder).named_parameters())
            + list(self.predictor.named_parameters())
            + [("mask_token", self.mask_token)]
        )

        param_to_module = {}
        all_modules = (
            list(self._unwrap(self.online_encoder).modules())
            + list(self.predictor.modules())
        )
        for m in all_modules:
            for pn, p in m.named_parameters(recurse=False):
                param_to_module[p] = m

        for pn, p in named_params:
            if not p.requires_grad:
                continue
            container = param_to_module.get(p, None)
            if pn.endswith("bias"):
                no_decay.add(pn)
            elif container is not None and isinstance(container, blacklist):
                no_decay.add(pn)
            elif any(x in pn for x in ["pos_embed", "cls_token", "mask_token"]):
                no_decay.add(pn)
            else:
                decay.add(pn)

        param_dict = {pn: p for pn, p in named_params if p.requires_grad}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)
                        if pn in param_dict],    "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)
                        if pn in param_dict],    "weight_decay": 0.0},
        ]

        logging.info(
            f"Optimiser: {len(decay)} params with WD, "
            f"{len(no_decay)} without. "
            f"Target encoder excluded (EMA only)."
        )
        return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

    def setup_scheduler(self, optimiser, warmup_epochs, total_epochs):
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=(total_epochs - warmup_epochs), eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )