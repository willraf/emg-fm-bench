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

from torch import dist

from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.tasks import setup_task

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

        self.test_loader = setup_data(cfg, mode="test", distributed=self.distributed)

        self.task = setup_task(cfg, device=self.device, distributed=self.distributed, is_main=self.is_main, save_dir=self.save_dir)

        if cfg.model.checkpoint is not None:
            checkpoint = self.task.load_checkpoint(cfg.model.checkpoint)
        else: 
            raise ValueError("No checkpoint specified for evaluation. Please provide a checkpoint path in the config file.")

        # Wrap after loading. Checkpointed models are saved in non-DDP format, so we need to load first and then wrap if distributed.
        if self.distributed:
            self.task.wrap_distributed(self.local_rank)

        self.metrics = []

    def evaluate(self):
        
        self.task.eval()

        for batch_idx, batch in enumerate(self.test_loader):
            metrics = self.task.eval_step(batch, batch_idx=batch_idx)
        
        metrics = self._reduce_metrics(metrics)
        logging.info(f"Test Metrics: " +
                " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        self._record_metrics(metrics)

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


    def _record_metrics(self, metrics):
        record = {
            **{f"test_{k}": v for k, v in metrics.items()}
        }
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
                # Add plotting here
                pass


def main(cfg, overwrite=False):

    evaluator = Evaluator(cfg, overwrite=overwrite)
    evaluator.evaluate()
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

