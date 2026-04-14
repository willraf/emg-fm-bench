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

Date:
    31/07/2025
"""

import argparse
import logging
import traceback

from omegaconf import OmegaConf

from src.utils.utils import add_log_file, load_config, get_experiment_directory, setup_logging, setup_distributed

from scripts.train import main as train
from scripts.eval import main as evaluate
from scripts.predict import main as predict

from scripts.train import Trainer

import hydra
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
import gc


def prevent_logging():
    logger = logging.getLogger()
    
    # 1. Clear any existing handlers (this nukes the auto-config from the warning)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.NullHandler())
    return

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    
    distributed, device, local_rank, rank = setup_distributed(ddp_debug_mode=cfg.experiment.get("ddp_debug_mode", False))
    if rank != 0:
        prevent_logging()

    experiment_dir = get_experiment_directory(cfg)
    add_log_file(experiment_dir / "main.log")
    logging.info(f"Experiment directory: {experiment_dir}")

    try:
        find_batch_size = cfg.experiment.find_batch_size
        min_batch_size = cfg.experiment.min_batch_size
        base_batch_size = cfg.experiment.base_batch_size
        base_learning_rate = cfg.experiment.base_learning_rate
        
        # (Optionally) Find max per-GPU batch size
        if find_batch_size:

            finder = Trainer(cfg, overwrite=True)
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            per_gpu_batch_size = finder.find_max_batch_size()

            # Gradient accumulation if per-GPU batch size is too small
            if per_gpu_batch_size < min_batch_size:
                accum_steps = (min_batch_size + per_gpu_batch_size - 1) // per_gpu_batch_size
                logging.warning(
                    f"Per-GPU batch size {per_gpu_batch_size} < minimum {min_batch_size}. "
                    f"Using {accum_steps} gradient accumulation steps."
                )
            else:
                accum_steps = 1

            cfg.data.batch_size = per_gpu_batch_size
            cfg.training.num_accumulation_steps = accum_steps   

            # Scale learning rate by effective batch size 
            effective_batch_size = per_gpu_batch_size * accum_steps * world_size
            scaled_lr = base_learning_rate * (effective_batch_size / base_batch_size)
            cfg.finetuning.learning_rate = scaled_lr
            logging.info(
                f"Effective batch size: {per_gpu_batch_size} (per-GPU) "
                f"× {accum_steps} (accum) × {world_size} (GPUs) = {effective_batch_size}. "
                f"LR scaled from {base_learning_rate} → {scaled_lr:.6f}."
            )

            # Save config after batch-size discovery, before training
            OmegaConf.save(cfg, experiment_dir / "config.yaml")

            # Clean up the finder trainer before building the real one
            del finder
            gc.collect()
            torch.cuda.empty_cache()

        else:
            OmegaConf.save(cfg, experiment_dir / "config.yaml")
        
        train_results = train(cfg, overwrite=True, find_batch_size=False)
        logging.info(f"Training completed with results: {train_results}")

        # Clear GPU memory before evaluation
        gc.collect()
        torch.cuda.empty_cache()

        if find_batch_size:
            eval_finder = Trainer(cfg, overwrite=False)
            eval_batch_size = eval_finder.find_max_batch_size()
            cfg.data.batch_size = eval_batch_size
            logging.info(f"Eval batch size: {eval_batch_size}")
            del eval_finder
            gc.collect()
            torch.cuda.empty_cache()

        eval_results = evaluate(cfg, overwrite=True)
        logging.info(f"Evaluation completed with results: {eval_results}")

    except Exception as e:
        # Capture the full traceback
        error_msg = traceback.format_exc()
        logging.error(f"Job failed with error:\n{error_msg}")
        try:
            with open(experiment_dir / "error.log", "w") as f:
                f.write(error_msg)
        except:
            pass
        # Re-raise for Hydra to acknowledge the failure
        raise e
    
    finally:
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

