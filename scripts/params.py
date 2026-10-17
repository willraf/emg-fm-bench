""" Module params.py

Script to get the number of active parameters of each model

Usage:

Date:
    30/04/2026
"""


import hydra
import logging

import torch
from src.utils.utils import get_experiment_directory, setup_distributed, save_exception, set_seed, setup_save_dir, prevent_logging

from scripts.train import Trainer

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    # 1. Set the random seed
    set_seed(1)

    # 2. Setup distributed training
    distributed, device, rank, local_rank = setup_distributed(ddp_debug_mode=False)
    is_main = (rank == 0)

    # 3. setup logging and results, config, figures etc. saving directory
    # Only the main process should log to console
    if not is_main:
        prevent_logging()

    try:
        trainer = Trainer(cfg, distributed=False, device=device, rank=rank, local_rank=local_rank, save_dir=None)
        active_count, inactive_count, _, _ = trainer.count_active_parameters()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise e

    logging.info(f"Model: {cfg.model.fm.name}, Dataset: {cfg.data.root_key}, Window Length: {cfg.data.window_length}, Freeze Backbone: {cfg.task.freeze_backbone}, From Scratch: {cfg.task.from_scratch}")
    logging.info(f"Active parameters: {active_count}")
    logging.info(f"Inactive parameters: {inactive_count}")

    del trainer
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()