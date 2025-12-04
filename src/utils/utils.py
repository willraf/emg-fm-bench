from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import shutil
import csv
import pandas as pd

from omegaconf import OmegaConf

from typing import Union, Optional, Callable, Dict
from torch.nn.parameter import Parameter
from typing import Iterator

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

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
    model_name = cfg.model.name

    if save_flag:
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"{timestamp}_{model_name}"
        experiment_dir = Path("experiments") / experiment_name
    else:
        experiment_dir = Path("experiments") / "TEMP"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def setup_logger(experiment_dir: Union[str, Path]): 
    """
    Setup logger for experiment
    """
    log_dir = Path(experiment_dir) / "logs"
    # This line should be redundant
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "experiment.log"),
            logging.StreamHandler()
        ]
    )


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


def setup_optimiser(cfg: OmegaConf, model_parameters: Iterator[Parameter]) -> optim.Optimizer:
    """
    Create optimiser from config file
    """
    optimizer_name = cfg.training.optimizer
    learning_rate = cfg.training.learning_rate
    momentum = cfg.training.momentum
    weight_decay = cfg.training.weight_decay

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model_parameters, 
            lr=learning_rate, 
            momentum=momentum
        )
        logging.info("Using SGD optimizer")
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model_parameters, 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logging.info("Using Adam optimizer")
    elif optimizer_name == 'adamw':
        optimizer = optim.Adam(
            model_parameters, 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logging.info("Using Adam optimizer")
    else:
        raise ValueError("Optimizer not recognised")

    return optimizer


def save_checkpoint(generator: nn.Module,
                    discriminator_stft: nn.Module,
                    discriminator_wave: nn.Module,
                    optimizer_G: optim.Optimizer,
                    optimizer_D: optim.Optimizer,
                    epoch: int,
                    val_loss: float,
                    checkpoint_path: Union[str, Path]):
    """
    Save the state of all models (Generator, Discriminators) and optimizers
    for the SoundStream GAN architecture.

    Args:
        generator (nn.Module): The Generator (Encoder, RVQ, Decoder).
        discriminator_stft (nn.Module): The STFT-based Discriminator.
        discriminator_wave (nn.Module): The Multi-scale Wave-based Discriminator.
        optimizer_G (optim.Optimizer): The optimizer for the Generator.
        optimizer_D (optim.Optimizer): The optimizer for the Discriminators.
        epoch (int): The current epoch number.
        val_loss (float): The validation loss at this epoch (often L_G total).
        checkpoint_path (Union[str, Path]): Path to save the checkpoint file.
    """
    
    # 1. Gather all state dictionaries
    checkpoint = {
        'epoch': epoch,
        'val_loss': val_loss,
        
        # Generator components
        'generator_state_dict': generator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        
        # Discriminator components (saved separately for clarity)
        'discriminator_stft_state_dict': discriminator_stft.state_dict(),
        'discriminator_wave_state_dict': discriminator_wave.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }

    path = Path(checkpoint_path)
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved: {path}")


def load_checkpoint(checkpoint_path: Union[str, Path], 
                     generator: nn.Module,
                     discriminator_stft: nn.Module,
                     discriminator_wave: nn.Module,
                     optimizer_G: optim.Optimizer = None,
                     optimizer_D: optim.Optimizer = None) -> Dict:
    """
    Load model and optimizer states from a checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        logging.warning(f"Checkpoint not found: {path}")
        return {}

    checkpoint = torch.load(path)

    # Load model states
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator_stft.load_state_dict(checkpoint['discriminator_stft_state_dict'])
    discriminator_wave.load_state_dict(checkpoint['discriminator_wave_state_dict'])
    
    # Load optimizer states (optional, needed only to resume training)
    if optimizer_G and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    if optimizer_D and 'optimizer_D_state_dict' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    logging.info(f"Checkpoint loaded successfully from {path} (Epoch {checkpoint['epoch']})")
    return checkpoint
def setup_save_dir(cfg: OmegaConf,
                    overwrite: bool = False, 
                    mode: str = "train"):
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
    # Create experiment directory
    experiment_dir = Path(get_experiment_directory(cfg))
    save_dir = experiment_dir / mode 

    # Handle existing training directory
    if save_dir.exists():
        logging.warning(f"Directory {save_dir} already exists.")

        if overwrite:
            logging.info("Overwriting existing directory.")
            shutil.rmtree(save_dir)
        else:
            response = input(f"Directory '{save_dir}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                logging.info("Aborting run.")
                exit(0)
            else:
                shutil.rmtree(save_dir)

    # Create necessary directories
    save_dir.mkdir(parents=True, exist_ok=False)
    (save_dir / "figures").mkdir(exist_ok=True)

    # Save logger to file
    change_log_file(save_dir / f"{mode}.log")

    # Save a copy of the config file
    OmegaConf.save(cfg, save_dir / "config.yaml")

    if mode == "train":

        (save_dir / "checkpoints").mkdir(exist_ok=True)

        # Create file to save training results
        with (save_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss"])
    
    return save_dir


## --------------------------------------------------------------
## -------------------------- Meta code -------------------------
## --------------------------------------------------------------
            
def get_contiguous_ones(binary_vector: np.ndarray) -> list[tuple[int, int]]:
    """Get a list of (start_idx, end_idx) for each contiguous block of True values."""
    if (binary_vector == 0).all():
        return []

    ones = np.where(binary_vector)[0]
    boundaries = np.where(np.diff(ones) != 1)[0]
    return [
        (ones[i], ones[j])
        for i, j in zip(
            np.insert(boundaries + 1, 0, 0), np.append(boundaries, len(ones) - 1)
        )
    ]


def get_ik_failures_mask(joint_angles: np.ndarray) -> np.ndarray:
    """Compute mask that is True where there are no ik failures."""
    zeros = np.zeros_like(joint_angles)  # (..., joint)
    is_zero = np.isclose(joint_angles, zeros)
    return ~np.all(is_zero, axis=-1)


## --------------------------------------------------------------