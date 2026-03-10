"""
Script eval

Handles model testing/evaluation.

Usage:
    $ python -m scripts.eval --config configs/eval.yaml
    $ python -m scripts.eval --config configs/eval.yaml --default configs/default.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import argparse
from pathlib import Path
import logging
import csv

from omegaconf import OmegaConf

from typing import Union, Optional, Callable

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.tasks import setup_task

import wandb

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Evaluator:
    
    def __init__(self, cfg, overwrite=False):
        self.cfg = cfg
        
        self.overwrite = overwrite        
        # Setup GPU device and distributed training
        self.distributed, self.device, self.rank, self.local_rank = setup_distributed()
        print(f"Process {self.rank} using device {self.device}, distributed={self.distributed}, local_rank={self.local_rank}")
        world_size = dist.get_world_size() if self.distributed else 1
        print(f"World size: {world_size}")
        self.is_main = (self.rank == 0)

        self.save_dir = setup_save_dir(cfg, overwrite=overwrite, mode='test', is_main_process=self.is_main)

        self.test_loader = setup_data(cfg.data, mode="test", distributed=self.distributed)

        self.task = setup_task(cfg, device=self.device, distributed=self.distributed, is_main=self.is_main, save_dir=self.save_dir)

        if cfg.model.checkpoint is not None:
            checkpoint = self.task.load_checkpoint(cfg.model.checkpoint)
        else: 
            raise ValueError("No checkpoint specified for evaluation. Please provide a checkpoint path in the config file.")

        # Wrap after loading. Checkpointed models are saved in non-DDP format, so we need to load first and then wrap if distributed.
        if self.distributed:
            self.task.wrap_distributed(self.local_rank)

        self.metrics = []

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
   

    def evaluate(self):

        if self.distributed:
            dist.barrier()  # Ensure all processes have finished training before validation

            if hasattr(self.test_loader.sampler, "set_epoch"):
                self.test_loader.dataset.set_epoch(0)  # If using a distributed sampler, set epoch for shuffling


        self.task.eval()

        total_loss = 0.0
        num_batches = len(self.test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                total_loss += self.task.eval_step(batch)
            
        avg_loss = total_loss / num_batches

        test_metrics = self.task.test_metrics.compute()
        self.task.test_metrics.reset()

        self._record_metrics(test_metrics)
        

    def _record_metrics(self, metrics):
        scalar_record = {}
        for k, v in metrics.items():
            if np.isscalar(v): # Only add numbers (loss, acc, f1)
                scalar_record[f"test_{k}"] = v

        self.metrics.append(scalar_record)
            
        if self.is_main:
            fpath = self.save_dir / "results.csv"

            write_header = not fpath.exists()
            with open(fpath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=scalar_record.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(scalar_record)

            if self.use_wandb:
                wandb_data = {**scalar_record}

                # if "confusion_matrix" in metrics:
                #     wandb_data["test/confusion_matrix"] = wandb.plot.confusion_matrix(
                #         probs=None,
                #         y_true=metrics["targets"],
                #         preds=metrics["preds"],
                #         class_names=self.cfg.data.get("class_names", None)
                #     )
                wandb.log(wandb_data)

            if self.cfg.experiment.plot_examples and "confusion_matrix" in metrics:
                vis.plot_confusion_matrix(metrics["confusion_matrix"], class_names=self.cfg.data.get("class_names", None), fpath=self.save_dir / "figures" / "confusion_matrix.png")



def main(cfg, overwrite=False):

    evaluator = Evaluator(cfg, overwrite=overwrite)
    evaluator.evaluate()

    if evaluator.is_main and evaluator.use_wandb:
        wandb.finish()

    if evaluator.distributed:
        dist.barrier()  # Ensure all processes have finished writing results before any process exits
        dist.destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    setup_logging()
    
    main(cfg, overwrite=args.overwrite)

