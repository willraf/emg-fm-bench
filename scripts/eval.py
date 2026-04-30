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
    
    def __init__(self, cfg, distributed, device, rank, local_rank, save_dir):
        self.cfg = cfg
        self.distributed = distributed
        self.device = device
        self.rank = rank
        self.local_rank = local_rank
        self.save_dir = save_dir

        checkpoint_path = cfg.model.checkpoint
        
        self.is_main = (self.rank == 0)


        self.test_loader = setup_data(cfg.data, mode="test", distributed=self.distributed)

        self.task = setup_task(cfg, device=self.device, distributed=self.distributed, is_main=self.is_main)

        if checkpoint_path is not None:
            _ = self.task.load_checkpoint(cfg.model.checkpoint)
        else: 
            raise ValueError("No checkpoint specified for evaluation. Please provide a checkpoint path in the config file.")

        # Wrap after loading. Checkpointed models are saved in non-DDP format, so we need to load first and then wrap if distributed.
        if self.distributed:
            self.task.wrap_distributed(self.local_rank)

        self.metrics = []

        self.use_wandb = (self.is_main and wandb.run is not None)
   

    def evaluate(self):

        if self.distributed:
            dist.barrier()  # Ensure all processes have finished training before validation

            if hasattr(self.test_loader.sampler, "set_epoch"):
                self.test_loader.sampler.set_epoch(0)  # If using a distributed sampler, set epoch for shuffling


        self.task.eval()

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

        if self.distributed:
            dist.barrier()
        
        test_metrics = self.task.test_metrics.compute()
        self.task.test_metrics.reset()

        if hasattr(self.task, "measure_inference_latency") and self.is_main:
            batch = next(iter(self.test_loader))
            mean_latency, std_latency = self.task.measure_inference_latency(batch, n_warmup=30, n_runs=200)

            test_metrics["inference_latency_mean"] = mean_latency
            test_metrics["inference_latency_std"] = std_latency

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
    

    def find_max_batch_size(self, start_bs=2, max_batch_size=256, safety_factor=0.8) -> int:
        recommended = torch.tensor(0, device=self.device)
        
        batch_size = start_bs
        last_good = 0

        while batch_size <= max_batch_size:
            if self.is_main:
                logging.info(f"Testing batch size {batch_size}...")
            
            # Ensure everyone uses the same batch size
            self.cfg.data.batch_size = batch_size
            
            loader = setup_data(self.cfg.data, mode="test", distributed=self.distributed)
            
            try:
                batch = next(iter(loader))
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                self.task.eval()  # Ensure model is in eval mode for batch size testing
                self.task.eval_step(batch)
            
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
                    logging.warning(f"Error at batch size {batch_size}: {str(e)}")
                    break #raise

        found = max(1, int(last_good * safety_factor))
        recommended = torch.tensor(found, device=self.device)

        if self.distributed:
            # Final sync to make sure all ranks agree on the result
            dist.all_reduce(recommended, op=dist.ReduceOp.MIN)
            dist.barrier()

        return int(recommended.item())
        


def main(cfg, overwrite=False, find_batch_size=False):
    """
    Main provides an entry point for evaluating a model based on a given configuration.
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
    
    eval_dir = setup_save_dir(cfg, experiment_dir=experiment_dir, overwrite=overwrite, mode='test', is_main_process=is_main)

    # 4. Setup weights and biases logging
    if is_main and use_wandb:
        wandb.init(
            project=wandb_project,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True), # Log full config
            dir=eval_dir # Keep W&B local logs in the experiment folder
        )

    try:
        evaluator = Evaluator(cfg, distributed, device, rank, local_rank, save_dir=eval_dir)

        if find_batch_size:
            batch_size = evaluator.find_max_batch_size()
            return batch_size

        evaluator.evaluate()

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