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

import wandb

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

        self.task = setup_task(cfg, device=self.device, distributed=self.distributed, is_main=self.is_main, save_dir=self.save_dir)

        self.train_loader = setup_data(cfg.data, mode="train", distributed=self.distributed)
        self.val_loader = setup_data(cfg.data, mode="val", distributed=self.distributed)

        self.epochs_run = 0
        if cfg.model.checkpoint is not None:
            checkpoint = self.task.load_checkpoint(cfg.model.checkpoint)
            self.epochs_run = checkpoint.get('epoch', 0)

        # For storing metric history across epochs
        self.metrics = []

        # Wrap after loading. Checkpointed models are saved in non-DDP format, so we need to load first and then wrap if distributed.
        if self.distributed:
            self.task.wrap_distributed(self.local_rank)

        # Gradient accumulation
        self.num_accumulation_steps = cfg.training.get("num_accumulation_steps", 1)

        # Initialise weights and biases
        if self.is_main and cfg.logging.get("use_wandb", False):
            wandb.init(
                project=cfg.logging.get("wandb_project", "emg-foundation-model"),
                name=cfg.experiment.name,
                config=OmegaConf.to_container(cfg, resolve=True), # Log full config
                dir=self.save_dir # Keep W&B local logs in your experiment folder
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
   

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

        for epoch in range(self.epochs_run, num_epochs):

            train_metrics = self._run_epoch(epoch)
            val_metrics = self._val(epoch)

            if hasattr(self.task, 'scheduler') and self.task.scheduler is not None:
                self.task.scheduler.step()
                current_lr = self.task.optimiser.param_groups[0]['lr']
                logging.info(f"End of Epoch {epoch} - Learning Rate: {current_lr:.6f}")

            # Early stopping and best model saving
            if self.is_main and patience is not None:

                val_loss = val_metrics['loss']

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

            if self.is_main:
                self._record_metrics(epoch, train_metrics, val_metrics, best_val_loss)

            if stop_training:
                logging.info(f"No improvement in validation loss for {patience} epochs. Early stopping triggered.")
                break

        logging.info("Training complete.")


    def _run_epoch(self, epoch):
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch) 

        self.task.train()
        self.task.train_metrics.reset()

        pbar = tqdm(
            self.train_loader, 
            disable=(not self.is_main), 
            desc=f"Epoch {epoch}",
            ncols=100
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Might be using gradient accumulation. Only update weights and log every num_accumulation_steps batches, or on the last batch
            is_update_step = ((batch_idx + 1) % self.num_accumulation_steps == 0) or (batch_idx == len(self.train_loader) - 1)
            loss = self.task.train_step(batch, is_update_step)

            if self.is_main:
                pbar.set_postfix({"loss": f"{loss:.4f}"})
        

        epoch_metrics = self.task.train_metrics.compute()
        self.task.train_metrics.reset()
        # Only the main process will have valid metrics when using distributed training, so we return None for other ranks
        return epoch_metrics



    @torch.no_grad()
    def _val(self, epoch):

        if self.distributed:
            dist.barrier()  # Ensure all processes have finished training before validation

            if hasattr(self.val_loader.sampler, "set_epoch"):
                self.val_loader.sampler.set_epoch(epoch)  # If using a distributed sampler, set epoch for shuffling


        self.task.eval()

        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                total_loss += self.task.val_step(batch)
            
        avg_loss = total_loss / num_batches

        val_metrics = self.task.val_metrics.compute()
        self.task.val_metrics.reset()
        
        return val_metrics
    

    def _record_metrics(self, epoch, train_metrics, val_metrics, best_val_loss):
        # 1. Create the scalar record for CSV (Numbers only)
        scalar_record = {"epoch": epoch, "best_val_loss": best_val_loss}
        
        for phase, metrics in [("train", train_metrics), ("val", val_metrics)]:
            if metrics is None: continue # Handling non-main ranks
            for k, v in metrics.items():
                if np.isscalar(v): # Only add numbers (loss, acc, f1)
                    scalar_record[f"{phase}_{k}"] = v

        # Keep track of metrics history for each epoch
        self.metrics.append(scalar_record)

        # Log results
        logging.info(f"Epoch {epoch} Train| " +
            " | ".join([f"{k}: {v:.4f}" for k, v in scalar_record.items()])
        )

        fpath = self.save_dir / "results.csv"

        write_header = not fpath.exists()
        with open(fpath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(scalar_record)

        if self.use_wandb:
            wandb_data = {**scalar_record}
            
            # Log Confusion Matrix as a specialized WandB Plot
            # if "confusion_matrix" in val_metrics:
            #     wandb_data["val/confusion_matrix"] = wandb.plot.confusion_matrix(
            #         probs=None,
            #         y_true=val_metrics["targets"],
            #         preds=val_metrics["preds"],
            #         class_names=self.cfg.data.get("class_names", None)
            #     )
            wandb.log(wandb_data, step=epoch)

        if self.cfg.experiment.plot_examples:
            train_losses = [m['train_loss'] for m in self.metrics]
            val_losses = [m['val_loss'] for m in self.metrics]
            epochs = [m['epoch'] for m in self.metrics]
            vis.plot_loss_curves(train_losses, val_losses, epochs, fpath=self.save_dir / "figures" / "loss_curves.png")


    def find_max_batch_size(self, start_bs=2, max_batch_size=256, safety_factor=0.8) -> int:
        """
        Utility function to find the maximum batch size that fits in GPU memory.
        Uses a binary search approach, starting from `start_bs` and doubling until OOM, then refining.

        Args:
            start_bs (int): Initial batch size to try
            max_batch_size (int): Maximum batch size to test
            safety_factor (float): Factor to reduce the found batch size by for safety margin
        Returns:
            int: Maximum batch size that fits in GPU memory with safety margin
        """
        recommended = torch.tensor(0, device=self.device)

        if self.is_main:
            logging.info("\n🔍 Starting batch size probe...\n")

            batch_size = start_bs
            last_good = start_bs

            while batch_size <= max_batch_size:

                logging.info(f"Testing batch size {batch_size}...")
                self.cfg.data.batch_size = batch_size
                loader = setup_data(self.cfg.data, mode="val")
                
                batch = next(iter(loader))

                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    self.task.train_step(batch)

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

            found = max(1, int(last_good * safety_factor))
            logging.info(f"\nRecommended per-GPU batch size: {found}")
            recommended = torch.tensor(found, device=self.device)

        # Sync GPUs
        if self.distributed:
            print(f'Process {self.rank} recommended batch size {recommended.item()}')
            dist.broadcast(recommended, src=0)
            print(f'After sync: Process {self.rank} recommended batch size {recommended.item()}')

        return int(recommended.item())
        

def main(cfg, overwrite=False, find_batch_size=False):
    trainer = Trainer(cfg, overwrite=overwrite)

    if find_batch_size:
        batch_size = trainer.find_max_batch_size()
        return batch_size

    trainer.train()

    if trainer.is_main and trainer.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--find_batch_size', action='store_true', required=False, default=False, help="Run batch size finder instead of training")
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    setup_logging()
    
    main(cfg, overwrite=args.overwrite, find_batch_size=args.find_batch_size)

    # Cleanup distributed properly
    if dist.is_initialized():
        dist.barrier()
        destroy_process_group()
