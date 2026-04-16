""" 
Script main

Main entry point for the full training/evaluation/prediction pipeline.

Usage:
    % python -m scripts.main --config configs/main.yaml
    % python -m scripts.main --config configs/main.yaml --default configs/default.yaml
    % python -m torch.distributed.run --nproc_per_node=1 -m scripts.main --config configs/main.yaml
    
    With Hydra:
    % python -m scripts.main --config-name other_config
    % python -m torch.distributed.run   --nproc_per_node=4 -m scripts.main --config-name other_config
    % python -m torch.distributed.run   --nproc_per_node=4 -m scripts.main +experiment=test

Authors:
    Will Raftery


TODO:
    - Change config to not use finetune section (will get confusing in the future)
    - Standardise responsibility of the entry point (main functions in main.py, train.py, eval.py)
        - Distributed setup
        - Logging
        - Wandb setup
        - Set seed
        - Finish wandb
        - Cleanup distributed

Date:
    31/07/2025
"""

import argparse
import logging
import wandb
import traceback

from omegaconf import OmegaConf

from src.utils.utils import get_experiment_directory, setup_distributed, get_hardware_info

from scripts.train import Trainer
from scripts.eval import Evaluator

import hydra
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
import gc
import datetime


def prevent_logging():
    logger = logging.getLogger()
    
    # Clear any existing handlers (this nukes the auto-config from the warning)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.NullHandler())
    return


def get_max_batch_size(cfg, mode="train"):
    """
    Use batch size probing to find the maximum batch size that fits in GPU memory for training or evaluation.
    """
    if mode == "train":
        finder = Trainer(cfg, overwrite=True)
    elif mode == "eval":
        finder = Evaluator(cfg, overwrite=True)
    else:
        raise ValueError(f"Invalid mode {mode} for find_batch_size. Must be 'train' or 'eval'.")
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_gpu_batch_size = finder.find_max_batch_size()
    logging.info(f"Found max batch size of {per_gpu_batch_size} per GPU in {mode} mode with world size {world_size}.")
    del finder
    gc.collect()
    torch.cuda.empty_cache()
    return per_gpu_batch_size


def gradient_accumulation(cfg, per_gpu_batch_size):
    """
    Determine the number of gradient accumulation steps needed to achieve the minimum effective batch size if the per-GPU batch size is too small.
    """
    min_batch_size = cfg.experiment.min_batch_size
    if per_gpu_batch_size < min_batch_size:
        accum_steps = (min_batch_size + per_gpu_batch_size - 1) // per_gpu_batch_size
        logging.warning(
            f"Per-GPU batch size {per_gpu_batch_size} < minimum {min_batch_size}. "
            f"Using {accum_steps} gradient accumulation steps."
        )
        return accum_steps
    return 1


def scale_learning_rate(cfg, per_gpu_batch_size, accum_steps):
    """
    Scale the learning rate according to the effective batch size if batch size discovery is used.
    """
    base_learning_rate = cfg.experiment.base_learning_rate
    base_batch_size = cfg.experiment.base_batch_size

    # Compute the effective batch size based on num GPUs and accumulation steps
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    effective_batch_size = per_gpu_batch_size * accum_steps * world_size

    scaled_lr = base_learning_rate * (effective_batch_size / base_batch_size)
    logging.info(
        f"Effective batch size: {per_gpu_batch_size} (per-GPU) "
        f"× {accum_steps} (accum) × {world_size} (GPUs) = {effective_batch_size}. "
        f"LR scaled from {base_learning_rate} → {scaled_lr:.6f}."
    )
    return scaled_lr


def save_exception(experiment_dir):
    error_msg = traceback.format_exc()
    logging.error(f"Job failed with error:\n{error_msg}")
    try:
        with open(experiment_dir / "error.log", "w") as f:
            f.write(error_msg)
    except:
        pass
    

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):

    find_batch_size = cfg.experiment.find_batch_size
    ddp_debug_mode = cfg.experiment.get("ddp_debug_mode", False)
    use_wandb = cfg.experiment.get("use_wandb", False)

    distributed, device, rank, local_rank = setup_distributed(ddp_debug_mode=ddp_debug_mode)
    is_main = (rank == 0)
    # Only the main process should log to console
    if rank != 0:
        prevent_logging()

    experiment_dir = get_experiment_directory(cfg)

    try: 
        if find_batch_size: 
            per_gpu_batch_size = get_max_batch_size(cfg, mode="train")
            accum_steps = gradient_accumulation(cfg, per_gpu_batch_size)
            scaled_lr = scale_learning_rate(cfg, per_gpu_batch_size, accum_steps)

            # Update config with discovered batch size and scaled learning rate
            cfg.data.batch_size = per_gpu_batch_size
            cfg.training.num_accumulation_steps = accum_steps
            cfg.finetuning.learning_rate = scaled_lr
        
        # Save config after batch-size discovery, before training
        OmegaConf.save(cfg, experiment_dir / "config.yaml")

        trainer = Trainer(cfg, overwrite=True)
        train_results = trainer.train()
        logging.info(f"Training completed with results: {train_results}")

        # Clear GPU memory before evaluation
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if find_batch_size:
            per_gpu_batch_size = get_max_batch_size(cfg, mode="eval")
            cfg.data.batch_size = per_gpu_batch_size

        evaluator = Evaluator(cfg, overwrite=True)
        eval_results = evaluator.evaluate()
        logging.info(f"Evaluation completed with results: {eval_results}")

    except Exception as e:
        # Capture the full traceback
        save_exception(experiment_dir)

        # Re-raise for Hydra to acknowledge the failure
        raise e

    finally:
        if is_main and use_wandb:
            wandb.finish()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            logging.info("Distributed process group destroyed.")

    return {"train_results": train_results, "eval_results": eval_results}
    

if __name__ == "__main__":
    main()
    # Cleanup distributed properly
    if dist.is_initialized():
        dist.barrier()
        destroy_process_group()

