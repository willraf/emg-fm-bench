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

    def __init__(self, cfg, distributed, device, rank, local_rank, save_dir):

        self.cfg = cfg
        self.distributed = distributed
        self.device = device
        self.rank = rank
        self.local_rank = local_rank
        self.save_dir = save_dir
        
        resume_training = cfg.experiment.resume
        checkpoint_path = cfg.model.checkpoint

        self.num_accumulation_steps = cfg.training.get("num_accumulation_steps", 1)

        self.is_main = (self.rank == 0)
        self.use_wandb = (self.is_main and wandb.run is not None)

        self.task = setup_task(cfg, device=self.device, distributed=self.distributed, is_main=self.is_main)

        self.train_loader = setup_data(cfg.data, mode="train", distributed=self.distributed)
        self.val_loader = setup_data(cfg.data, mode="val", distributed=self.distributed)

        self.epochs_run = 0
        self.prev_train_time = 0
        if checkpoint_path is not None and resume_training:
            checkpoint = self.task.load_checkpoint(checkpoint_path)
            self.epochs_run = checkpoint.get('epoch', 0) + 1
            self.prev_train_time = checkpoint.get('train_time', 0)

        # For storing metric history across epochs
        self.metrics = []

        # Wrap after loading. Checkpointed models are saved in non-DDP format, so we need to load first and then wrap if distributed.
        if self.distributed:
            self.task.wrap_distributed(self.local_rank)        
   

    def train(self):
        """
        Main training loop
        """
        num_epochs = self.cfg.training.epochs
        patience = self.cfg.training.patience

        best_val_loss = float('inf')
        epochs_no_improve = 0
        stop_training = False

        start_time = time.time()

        for epoch in range(self.epochs_run, num_epochs):

            train_metrics = self._run_epoch(epoch)
            val_metrics = self._val(epoch)

            # Calculate the cumulative training time up to this epoch, including any previous epochs if resuming from a checkpoint
            cumulative_time = time.time() - start_time + self.prev_train_time

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
                    self.task.save_checkpoint(checkpoint_path=checkpoint_path, epoch=epoch, val_loss=val_loss, train_time=cumulative_time)
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    stop_training = True

            if self.distributed:
                stop_tensor = torch.tensor(int(stop_training), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if self.is_main:
                self._record_metrics(epoch, train_metrics, val_metrics, best_val_loss, train_time=cumulative_time)

            if stop_training:
                logging.info(f"No improvement in validation loss for {patience} epochs. Early stopping triggered.")
                break

        logging.info("Training complete.")
        
        return best_val_loss if self.is_main else None

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
        
        if self.distributed:
            dist.barrier()

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

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                loss = self.task.val_step(batch)

        if self.distributed:
            dist.barrier()

        val_metrics = self.task.val_metrics.compute()
        self.task.val_metrics.reset()
        
        return val_metrics
    

    def _record_metrics(self, epoch, train_metrics, val_metrics, best_val_loss, train_time):
        # Create the scalar record for CSV (Numbers only)
        scalar_record = {"epoch": epoch, "best_val_loss": best_val_loss, "train_time": train_time}
        
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
        recommended = torch.tensor(0, device=self.device)
        
        batch_size = start_bs
        last_good = 0

        while batch_size <= max_batch_size:
            if self.is_main:
                logging.info(f"Testing batch size {batch_size}...")
            
            # Ensure everyone uses the same batch size
            self.cfg.data.batch_size = batch_size
            
            loader = setup_data(self.cfg.data, mode="train", distributed=self.distributed)
            
            try:
                batch = next(iter(loader))
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                self.task.train()  # Ensure model is in train mode for batch size testing, since that's usually more memory intensive than eval

                # Run a few training steps to get an accurate measure of memory usage
                for accum_step in range(3 * self.num_accumulation_steps):
                    is_update_step = ((accum_step + 1) % self.num_accumulation_steps == 0)
                    self.task.train_step(batch, is_update_step)

                if self.is_main:
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    logging.info(f"Success | Peak memory {peak:.2f} GB")
                
                last_good = batch_size
                batch_size *= 2

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.is_main: logging.info(f"OOM at {batch_size}")
                    break
                elif "received data size doesn't match expected size" in str(e).lower():
                    if self.is_main: logging.warning(f"Data size mismatch at {batch_size}. This can happen in distributed training when one rank OOMs but others don't. Assuming OOM and stopping.")
                    break
                else:
                    print(str(e).lower())
                    raise

        found = max(1, int(last_good * safety_factor))
        recommended = torch.tensor(found, device=self.device)

        if self.distributed:
            # Final sync to make sure all ranks agree on the result
            dist.all_reduce(recommended, op=dist.ReduceOp.MAX)
            dist.barrier()

        return int(recommended.item())
        
    def count_active_parameters(self):
        """
        Count parameters that are actually used in the forward pass
        by registering hooks on all parameters.
        """
        sample_batch = next(iter(self.train_loader))
        return self.task.count_active_parameters(sample_batch)
    

def main(cfg, overwrite=False, find_batch_size=False):
    """
    Main provides an entry point for training a model based on a given configuration.
    It handles:
    - Set random seed
    - Distributed
    - Logging & wandb
    - Setting up directory structures and saving
    - Trainer init and running
    - Tear down
    """
    # Experiment level setup
    seed = cfg.experiment.random_seed
    experiment_name = cfg.experiment.name
    save_experiment = cfg.experiment.save
    ddp_debug_mode = cfg.experiment.get("ddp_debug_mode", False)
    use_wandb = cfg.experiment.get("use_wandb", False)
    wandb_project = cfg.experiment.get("wandb_project", "emg-foundation-model")

    # 1. Set the random seed
    set_seed(seed)

    # 2. Setup distributed training
    distributed, device, rank, local_rank = setup_distributed(ddp_debug_mode=ddp_debug_mode)
    is_main = (rank == 0)

    # 3. setup logging and results, config, figures etc. saving directory
    experiment_dir = get_experiment_directory(experiment_name=experiment_name, save_flag=save_experiment)
    setup_logging(experiment_dir=experiment_dir, is_main_process=is_main)
    
    train_dir = setup_save_dir(cfg, experiment_dir=experiment_dir, overwrite=overwrite, mode='train', is_main_process=is_main)

    # 4. Setup weights and biases logging
    if is_main and use_wandb:
        wandb.init(
            project=wandb_project,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True), # Log full config
            dir=train_dir # Keep W&B local logs in the experiment folder
        )

    try:
        trainer = Trainer(cfg, distributed, device, rank, local_rank, save_dir=train_dir)

        if find_batch_size:
            batch_size = trainer.find_max_batch_size()
            return batch_size

        trainer.train()

    except Exception as e:
        # Capture the full traceback
        save_exception(experiment_dir)

        # Re-raise for Hydra to acknowledge the failure
        raise e

    finally:
        # Cleanup
        if is_main and use_wandb:
            wandb.finish()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            logging.info("Distributed process group destroyed.")
            

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--find_batch_size', action='store_true', required=False, default=False, help="Run batch size finder instead of training")
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    main(cfg, overwrite=args.overwrite, find_batch_size=args.find_batch_size)
