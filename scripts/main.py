""" 
Script main

Main entry point for the full training/evaluation/prediction pipeline.

Usage:
    % python -m scripts.main --config configs/main.yaml
    % python -m scripts.main --config configs/main.yaml --default configs/default.yaml
    % python -m torch.distributed.run --nproc_per_node=1 -m scripts.main --config configs/main.yaml
    
    With Hydra:
    % python -m scripts.main --config-name other_config
    % python -m scripts.main +experiment=other_experiment
    % python -m torch.distributed.run   --nproc_per_node=4 -m scripts.main --config-name other_config
    % python -m torch.distributed.run   --nproc_per_node=4 -m scripts.main +experiment=test

Date:
    31/07/2025
"""

import argparse
import logging
import wandb

from omegaconf import OmegaConf

from src.utils.utils import get_experiment_directory, setup_distributed, save_exception, set_seed, setup_save_dir, prevent_logging

from scripts.train import Trainer
from scripts.eval import Evaluator

import hydra
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
import gc
import datetime





def get_max_batch_size(cfg, distributed, device, rank, local_rank, mode="train"):
    """
    Use batch size probing to find the maximum batch size that fits in GPU memory for training or evaluation.
    """
    if mode == "train":
        finder = Trainer(cfg, distributed, device, rank, local_rank, save_dir=None)
    elif mode == "eval":
        finder = Evaluator(cfg, distributed, device, rank, local_rank, save_dir=None)
    else:
        raise ValueError(f"Invalid mode {mode} for find_batch_size. Must be 'train' or 'eval'.")
    
    max_batch_size = cfg.batch_size_finder.max_batch_size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    per_gpu_batch_size = finder.find_max_batch_size(max_batch_size=max_batch_size)
    logging.info(f"Found max batch size of {per_gpu_batch_size} per GPU in {mode} mode with world size {world_size}.")
    del finder
    gc.collect()
    torch.cuda.empty_cache()
    return per_gpu_batch_size


def gradient_accumulation(cfg, per_gpu_batch_size):
    """
    Determine the number of gradient accumulation steps needed to achieve the minimum effective batch size if the per-GPU batch size is too small.
    """
    min_batch_size = cfg.batch_size_finder.min_batch_size
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
    base_learning_rate = cfg.batch_size_finder.base_learning_rate
    base_batch_size = cfg.batch_size_finder.base_batch_size

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
    

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    """
    Main provides an entry point for the full pipeline.
    It handles:
    - Set random seed
    - Distributed
    - Logging & wandb
    - Setting up directory structures and saving
    - Trainer init, batch size discovery, and running
    - Evaluator init, batch size discovery, and running
    - Tear down
    """
    
    # Experiment level setup
    seed = cfg.experiment.random_seed
    experiment_name = cfg.experiment.name
    save_experiment = cfg.experiment.save
    find_batch_size = cfg.experiment.find_batch_size
    ddp_debug_mode = cfg.experiment.get("ddp_debug_mode", False)

    use_wandb = cfg.experiment.get("use_wandb", False)
    wandb_project = cfg.experiment.get("wandb_project", "emg-foundation-model")

    # 1. Set the random seed
    set_seed(seed)

    # 2. Setup distributed training
    distributed, device, rank, local_rank = setup_distributed(ddp_debug_mode=ddp_debug_mode)
    is_main = (rank == 0)
    
    logging.info(f"Rank {rank}, Local Rank {local_rank}, Device {device}, Distributed {distributed}")

    # 3. setup logging and results, config, figures etc. saving directory
    # Only the main process should log to console
    if not is_main:
        prevent_logging()

    experiment_dir = get_experiment_directory(experiment_name=experiment_name, save_flag=save_experiment)
    train_dir = setup_save_dir(cfg, experiment_dir=experiment_dir, overwrite=True, mode='train', is_main_process=is_main)
    eval_dir = setup_save_dir(cfg, experiment_dir=experiment_dir, overwrite=True, mode='eval', is_main_process=is_main)



     # 4. Setup weights and biases logging
    if is_main and use_wandb:
        wandb.init(
            project=wandb_project,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True), # Log full config
            dir=experiment_dir # Keep W&B local logs in the experiment folder
        )
    
    try: 
        if find_batch_size: 
            per_gpu_batch_size = get_max_batch_size(cfg, distributed, device, rank, local_rank, mode="train")
            accum_steps = gradient_accumulation(cfg, per_gpu_batch_size)
            scaled_lr = scale_learning_rate(cfg, per_gpu_batch_size, accum_steps)

            # Update config with discovered batch size and scaled learning rate
            cfg.data.batch_size = per_gpu_batch_size
            cfg.training.num_accumulation_steps = accum_steps
            cfg.task.learning_rate = scaled_lr
        
        # Save config after batch-size discovery, before training
        OmegaConf.save(cfg, experiment_dir / "config.yaml")

        trainer = Trainer(cfg, distributed, device, rank, local_rank, save_dir=train_dir)
        train_results = trainer.train()
        logging.info(f"Training completed with results: {train_results}")

        # Clear GPU memory before evaluation
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        if find_batch_size:
            per_gpu_batch_size = get_max_batch_size(cfg, distributed, device, rank, local_rank, mode="eval")
            cfg.data.batch_size = per_gpu_batch_size

        evaluator = Evaluator(cfg, distributed, device, rank, local_rank, save_dir=eval_dir)
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

