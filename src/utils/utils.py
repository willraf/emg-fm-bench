"""
Contains utility functions for setting up distributed training, logging, and experiment management.
"""


from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import random
import torch
import shutil
import csv

from omegaconf import OmegaConf

from typing import Union, Optional, Callable, Dict

import torch.optim as optim
import torch.nn as nn

import os
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist


def setup_distributed():
    """
    This funciton is exclusively for debugging distributed training on a single GPU

    Returns:
        distributed (bool)
        device (torch.device)
        rank (int)
        local_rank (int)
    """

    logging.warning("Using setup_distributed() from utils.py. This is only intended for debugging distributed training on a single GPU. For proper distributed training, use the setup_distributed() function in src/utils/distributed.py which uses torchrun for initialization.")

    # Check if already initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        gpu_count = torch.cuda.device_count()
        # Map local_rank to an available device ID (modulo logic)
        device_id = local_rank % gpu_count
        device = torch.device(f"cuda:{device_id}")
        
        return True, device, rank, local_rank

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # --- Distributed mode ---
    if world_size > 1:
        # Check how many GPUs are actually visible
        gpu_count = torch.cuda.device_count()
        # Map local_rank to an available device ID (modulo logic)
        device_id = local_rank % gpu_count
        torch.cuda.set_device(device_id)
        init_process_group(backend="gloo")

        device = torch.device(f"cuda:{device_id}")
        # device = torch.device("cpu")  # Force CPU for debugging

        return True, device, rank, local_rank

    # --- Single GPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        return False, device, 0, 0

    # --- CPU ---
    device = torch.device("cpu")
    return False, device, 0, 0


# def setup_distributed(ddp_debug_mode: bool = False):
#     """
#     Setup distributed training environment.

#     Returns:
#         distributed (bool)
#         device (torch.device)
#         rank (int)
#         local_rank (int)
#     """
#     if ddp_debug_mode:
#         return setup_distributed_debug()

#     # Check if already initialized
#     if dist.is_initialized():
#         rank = dist.get_rank()
#         world_size = dist.get_world_size()
#         local_rank = int(os.environ.get("LOCAL_RANK", 0))
#         device = torch.device(f"cuda:{local_rank}")
        
#         return True, device, rank, local_rank

#     rank = int(os.environ.get("RANK", 0))
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     world_size = int(os.environ.get("WORLD_SIZE", 1))

#     # --- Distributed mode ---
#     if world_size > 1:
#         torch.cuda.set_device(local_rank)
#         init_process_group(backend="nccl")

#         device = torch.device(f"cuda:{local_rank}")

#         return True, device, rank, local_rank

#     # --- Single GPU ---
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#         return False, device, 0, 0

#     # --- CPU ---
#     device = torch.device("cpu")
#     return False, device, 0, 0


def load_config(user_path: Path, default_path: Path) -> OmegaConf:
    """
    Load and merge user and default configuration files.

    Args:
        user_path (Path): Path to the user configuration file.
        default_path (Path): Path to the default configuration file.

    Returns:
        OmegaConf: Merged configuration object.
    """
    user_config = OmegaConf.load(user_path)
    default_config = OmegaConf.load(default_path)
    merged_config = OmegaConf.merge(default_config, user_config)
    return merged_config


def set_seed(seed: int):
    """
    Set seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_experiment_directory(cfg: OmegaConf) -> Path:
    """Create/get a directory for the experiment with a structured naming convention.
    
    Args:
        config (Config): Configuration object

    Returns:
        Path: Path to the experiment directory
    """
    save_flag = cfg.experiment.save
    experiment_name = cfg.experiment.name

    if save_flag:
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"{timestamp}"
        experiment_dir = Path("experiments") / experiment_name
    else:
        experiment_dir = Path("experiments") / "TEMP"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def setup_logging(experiment_dir: Optional[Union[str, Path]] = None, is_main_process: bool = True): 
    # Get the root logger
    root = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    if not is_main_process:
        # Set non-main ranks to only show Errors
        root.setLevel(logging.ERROR)
        root.addHandler(logging.NullHandler())
        return
    
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if experiment_dir:
        log_dir = Path(experiment_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "experiment.log")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.info("Logging initialized successfully.")


def change_log_file(log_fpath: Union[str, Path]):

    logger = logging.getLogger()  # Get the root logger
    
    # Remove existing file handlers
    for handler in logger.handlers[:]:  
        if isinstance(handler, logging.FileHandler):  
            logger.removeHandler(handler)
            handler.close()

    # Add new file handler
    file_handler = logging.FileHandler(log_fpath)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"Log file changed to: {log_fpath}")


def add_log_file(log_fpath: Union[str, Path], is_main_process: bool = True):

    if not is_main_process:
        return
    
    logger = logging.getLogger()  # Get the root logger

    # Add new file handler
    file_handler = logging.FileHandler(log_fpath)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"Log file added: {log_fpath}")


def setup_save_dir(cfg: OmegaConf,
                    overwrite: bool = False, 
                    mode: str = "train",
                    is_main_process: bool = True) -> Path:
    """
    Setup model and results saving

    If save is False, results are still saved to a TEMP directory and will be overwritten each run

    Args:
        cfg (OmegaConf): Configuration object
        overwrite (bool): Whether to overwrite existing directories
        mode (str): Mode of operation, either "train" "eval" or "predict"

    Returns:
        save_dir (Path): Path to the directory where results will be saved
    """
    if not is_main_process:
        return None
    
    # Create experiment directory
    experiment_dir = Path(get_experiment_directory(cfg))
    save_dir = experiment_dir / mode 

    # Check if we are resuming from a checkpoint
    is_resuming = cfg.model.get("checkpoint") is not None

    # Handle existing training directory
    if save_dir.exists():
        logging.warning(f"Directory {save_dir} already exists.")
        if is_resuming:
            logging.info(f"Resuming training in existing directory: {save_dir}")
            # We don't delete; we just ensure sub-dirs exist
        elif overwrite:
            logging.warning("Overwriting existing directory.")
            shutil.rmtree(save_dir)
        else:
            response = input(f"Directory '{save_dir}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                logging.info("Aborting run.")
                exit(0)
            else:
                shutil.rmtree(save_dir)

    # Create necessary directories
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "figures").mkdir(exist_ok=True)

    # Save logger to file
    add_log_file(save_dir / f"{mode}.log")

    # Save the resolved configuration to the experiment directory for reproducibility
    resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)
    with open(save_dir / "config.yaml", "w") as f:
        f.write(resolved_cfg)


    if mode == "train":

        (save_dir / "checkpoints").mkdir(exist_ok=True)
    
    return save_dir

