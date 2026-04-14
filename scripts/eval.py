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
from tqdm import tqdm

from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.tasks import setup_task

import wandb

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
                self.test_loader.sampler.set_epoch(0)  # If using a distributed sampler, set epoch for shuffling


        self.task.eval()

        total_loss = 0.0
        num_batches = len(self.test_loader)


        pbar = tqdm(
            self.test_loader, 
            disable=(not self.is_main), 
            ncols=100
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                loss = self.task.eval_step(batch)

                if self.is_main:
                    pbar.set_postfix({"loss": f"{loss:.4f}"})
            
        test_metrics = self.task.test_metrics.compute()
        self.task.test_metrics.reset()

        if self.is_main:
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

    # def find_max_batch_size(self, start_bs=2, max_batch_size=1024, safety_factor=0.8) -> int:
    #     """
    #     Utility function to find the maximum batch size that fits in GPU memory.
    #     Uses a binary search approach, starting from `start_bs` and doubling until OOM, then refining.

    #     Args:
    #         start_bs (int): Initial batch size to try
    #         max_batch_size (int): Maximum batch size to test
    #         safety_factor (float): Factor to reduce the found batch size by for safety margin
    #     Returns:
    #         int: Maximum batch size that fits in GPU memory with safety margin
    #     """
    #     recommended = 0

    #     if self.is_main:
    #         logging.info("\n🔍 Starting batch size probe...\n")

    #         batch_size = start_bs
    #         last_good = start_bs
    #         trial = 0

    #         while batch_size <= max_batch_size:

    #             logging.info(f"Trial {trial+1}: Testing batch size {batch_size}...")
    #             self.cfg.data.batch_size = batch_size
    #             loader = setup_data(self.cfg.data, mode="test")
                
    #             batch = next(iter(loader))

    #             try:
    #                 torch.cuda.empty_cache()
    #                 torch.cuda.reset_peak_memory_stats()

    #                 self.task.eval_step(batch)

    #                 peak = torch.cuda.max_memory_allocated() / 1e9
    #                 logging.info(f"Success | Peak memory {peak:.2f} GB")

    #                 last_good = batch_size
    #                 batch_size *= 2
    #                 trial += 1

    #             except RuntimeError as e:

    #                 if "out of memory" in str(e).lower():
    #                     logging.info(f"OOM at batch size {batch_size}")
    #                     break
    #                 else:
    #                     raise

    #         recommended = max(1, int(last_good * safety_factor))
    #         logging.info(f"\nRecommended per-GPU batch size: {recommended}")

    #     # Sync GPUs
    #     if self.distributed:
    #         bs_tensor = torch.tensor(recommended, device=self.device)
    #         dist.broadcast(bs_tensor, src=0)
    #         recommended = int(bs_tensor.item())

    #     return recommended
    

    def find_max_batch_size(self, start_bs=2, max_batch_size=256, safety_factor=0.8) -> int:
        recommended = torch.tensor(0, device=self.device)
        
        # EVERYONE runs this loop
        batch_size = start_bs
        last_good = 0

        while batch_size <= max_batch_size:
            if self.is_main:
                logging.info(f"Testing batch size {batch_size}...")
            
            # Ensure everyone uses the same batch size
            self.cfg.data.batch_size = batch_size
            
            # CRITICAL: Use distributed=True in the probe loader 
            # so DDP hooks are properly initialized
            loader = setup_data(self.cfg.data, mode="val", distributed=self.distributed)
            
            try:
                batch = next(iter(loader))
                torch.cuda.empty_cache()

                # Everyone calls this, so everyone hits .backward() together
                self.task.eval_step(batch)
                
                last_good = batch_size
                batch_size *= 2
                
                # Sync here to ensure no rank OOMs while others think it succeeded
                if self.distributed:
                    dist.barrier() 

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.is_main: logging.info(f"OOM at {batch_size}")
                    break
                else:
                    raise

        found = max(1, int(last_good * safety_factor))
        recommended = torch.tensor(found, device=self.device)

        if self.distributed:
            # Final sync to make sure all ranks agree on the result
            dist.all_reduce(recommended, op=dist.ReduceOp.MAX)
            dist.barrier()

        return int(recommended.item())
        


def main(cfg, overwrite=False, find_batch_size=False):

    evaluator = Evaluator(cfg, overwrite=overwrite)

    if find_batch_size:
        batch_size = evaluator.find_max_batch_size()

        # Cleanup properly before returning to main.py
        if evaluator.distributed:
            dist.barrier()
            destroy_process_group()
        return batch_size

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
    parser.add_argument('--find_batch_size', action='store_true', required=False, default=False, help="Run batch size finder instead of training")
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)

    distributed, device, local_rank, rank = setup_distributed()
    setup_logging(is_main_process=(rank == 0))

    main(cfg, overwrite=args.overwrite, find_batch_size=args.find_batch_size)
    # Cleanup distributed properly
    if dist.is_initialized():
        dist.barrier()
        destroy_process_group()