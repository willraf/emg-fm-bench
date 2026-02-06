"""
Script train

Handles model training and validation.

Usage:
    $ python -m scripts.train --config configs.train.yaml
    $ python -m scripts.train --config configs.train.yaml --default configs/default.yaml
    For distributed training:
    $ python -m torch.distributed.run   --nproc_per_node=1 -m scripts.train --config configs/main.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import argparse
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import time
from pathlib import Path
import csv
import os
from collections import defaultdict


# TODO remove generic utils import *
from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.tasks import setup_task

import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class Trainer:
    """
    Trainer class for (potentially distributed) training.
    Responsible for training, validation, checkpointing, and metric logging.

    Attributes:
        cfg (OmegaConf): Configuration object
        overwrite (bool): Whether to overwrite existing experiment directory
        distributed (bool): Whether distributed training is enabled
        device (torch.device): Device to run training on ("cpu" , "cuda", "cuda:<local_rank>")
        rank (int): Global rank of the current process
        local_rank (int): Local rank of the current process
        is_main (bool): Whether this is the main process (rank 0)
        save_dir (Path): Directory to save checkpoints and results
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        model (torch.nn.Module): Model to be trained
        optimiser (torch.optim.Optimizer): Optimiser for training
        loss_fn (callable): Loss function
        scheduler (optional): Curriculum scheduler
             For changing data difficulty over epochs, usually for synthetic data.
        epochs_run (int): Number of epochs already run (for resuming training)
        metrics (list): List of dicts recording training metrics per epoch
    """

    def __init__(self, cfg, overwrite=False):

        set_seed(cfg.experiment.random_seed)

        self.cfg = cfg
        self.overwrite = overwrite        
        # Setup GPU device and distributed training
        self.distributed, self.device, self.rank, self.local_rank = setup_distributed()
        print(f"Process {self.rank} using device {self.device}, distributed={self.distributed}, local_rank={self.local_rank}")
        world_size = dist.get_world_size() if self.distributed else 1
        print(f"World size: {world_size}")
        self.is_main = (self.rank == 0)

        self.save_dir = setup_save_dir(cfg, overwrite=overwrite, mode='train', is_main_process=self.is_main)

        self.train_loader = setup_data(cfg, mode="train", distributed=self.distributed)
        self.val_loader = setup_data(cfg, mode="val", distributed=self.distributed)

        self.task = setup_task(cfg, device=self.device, distributed=self.distributed, is_main=self.is_main, save_dir=self.save_dir)

        self.epochs_run = 0
        if cfg.model.checkpoint is not None:
            checkpoint = self.task.load_checkpoint(cfg.model.checkpoint)
            self.epochs_run = checkpoint['epoch']

        # Wrap after loading. Checkpointed models are saved in non-DDP format, so we need to load first and then wrap if distributed.
        if self.distributed:
            self.task.wrap_distributed(self.local_rank)

        self.metrics = []
   

    def train(self):
        """
        Main training loop
        """
        num_epochs = self.cfg.training.epochs
        patience = self.cfg.training.patience
        log_interval = self.cfg.logging.log_interval
        plot_examples_flag = self.cfg.experiment.plot_examples

        best_val_loss = float('inf')
        epochs_no_improve = 0
        stop_training = False

        self.task.train()

        for epoch in range(self.epochs_run, num_epochs):

            train_metrics = self._run_epoch(epoch)
            val_metrics = self._val(epoch)

            train_metrics = self._reduce_metrics(train_metrics)
            val_metrics = self._reduce_metrics(val_metrics)


            logging.info(f"Epoch {epoch} Train| " +
                " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            )

            logging.info(
                f"Epoch {epoch} Validation| " +
                " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            )

            self._record_metrics(epoch, train_metrics, val_metrics, best_val_loss)

            val_loss = val_metrics[self.task.monitor_metric]

            # Early stopping and best model saving
            if self.is_main:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                
                    checkpoint_path = self.save_dir / "checkpoints" / "best_model.pt"
                    self.task.save_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch, val_loss=val_loss)
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    stop_training = True

            if self.distributed:
                stop_tensor = torch.tensor(int(stop_training), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break


        logging.info("Training complete.")


    def _run_epoch(self, epoch):

        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)

        epoch_metrics = defaultdict(float)
        
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            batch_metrics = self.task.train_step(batch)

            for k, v in batch_metrics.items():
                epoch_metrics[k] += v

        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return dict(epoch_metrics)


    @torch.no_grad()
    def _val(self, epoch):

        if self.distributed:
            dist.barrier()  # Ensure all processes have finished training before validation

            if hasattr(self.val_loader.sampler, "set_epoch"):
                self.val_loader.dataset.set_epoch(epoch)  # If using a distributed sampler, set epoch for shuffling

        

        self.task.eval()

        epoch_metrics = defaultdict(float)
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):

                batch_metrics = self.task.val_step(batch, batch_idx, epoch)

                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v

        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        return dict(epoch_metrics)
    

    def _record_metrics(self, epoch, train_metrics, val_metrics, best_val_loss):
        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "best_val_loss": best_val_loss        }
        self.metrics.append(record)

        if self.is_main:
            fpath = self.save_dir / "results.csv"

            write_header = not fpath.exists()
            with open(fpath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

            if self.cfg.experiment.plot_examples:
                train_losses = [m['train_loss'] for m in self.metrics]
                val_losses = [m['val_loss'] for m in self.metrics]
                epochs = [m['epoch'] for m in self.metrics]
                vis.plot_loss_curves(train_losses, val_losses, epochs, fpath=self.save_dir / "figures" / "loss_curves.png")


    def _reduce_metrics(self, metrics: dict):
        """Reduce a dict of metrics across all processes by averaging. Only call this on the main process after reduction."""
        if not self.distributed:
            return metrics

        reduced = {}
        for k, v in metrics.items():
            t = torch.tensor(v, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            reduced[k] = (t / dist.get_world_size()).item()
        return reduced
        

    def find_max_batch_size(self, start_bs=32, max_trials=10, safety_factor=0.8) -> int:
        """
        Utility function to find the maximum batch size that fits in GPU memory.
        Uses a binary search approach, starting from `start_bs` and doubling until OOM, then refining.

        Args:
            start_bs (int): Initial batch size to try
            max_trials (int): Maximum number of trials to find the optimal batch size
            safety_factor (float): Factor to reduce the found batch size by for safety margin
        Returns:
            int: Maximum batch size that fits in GPU memory with safety margin
        """

        if not self.is_main:
            return None

        logging.info("\n🔍 Starting batch size probe...\n")

        batch_size = start_bs
        last_good = start_bs

        for trial in range(max_trials):
            logging.info(f"Trial {trial+1}/{max_trials}: Testing batch size {batch_size}...")
            cfg.data.batch_size = batch_size
            loader = setup_data(cfg, mode="val")
            
            batch = next(iter(loader))

            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                self.task.training_step(batch)

                peak = torch.cuda.max_memory_allocated() / 1e9
                logging.info(f"Success | Peak memory {peak:.2f} GB")

                last_good = batch_size
                batch_size *= 2

            except RuntimeError as e:

                if "out of memory" in str(e).lower():
                    logging.info(f"OOM at batch size {batch_size}")
                    break
                else:
                    raise

        recommended = max(1, int(last_good * safety_factor))
        logging.info(f"\nRecommended per-GPU batch size: {recommended}")

        return recommended
    

def main(cfg, overwrite=False):
    
    trainer = Trainer(cfg, overwrite=overwrite)
    # trainer.find_max_batch_size()
    trainer.train()
    if trainer.distributed:
        destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    setup_logging()
    
    main(cfg, overwrite=args.overwrite)
