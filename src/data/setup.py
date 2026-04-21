""" Module setup

Provides functions to set up the dataset and dataloaders automatically

Usage:
    from src.data.setup import setup_data
    from src.data import setup_data

    dataloaders, scheduler = setup_data(cfg, mode="train")

Authors:
    Will Raftery

Date:
    31/07/2025
"""

import importlib
from pathlib import Path
import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from omegaconf import OmegaConf
import logging
import inspect

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "src.data." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        print(f"Could not find a class that matches {target_dataset_name} in {dataset_filename}.py.")
        raise NotImplementedError(f'In {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matches {target_dataset_name} in lowercase.')

    return dataset


def setup_data(data_cfg: OmegaConf, mode: str = "train", distributed: bool = False) -> DataLoader:
    """
    Setup data loaders and scheduler based on the configuration.
    
    Args:
        cfg (OmegaConf): Configuration object
        mode (str): One of ["train", "val", "test", "predict"]
        distributed (bool): Whether to use DistributedSampler for distributed training

    Returns:
        dataloader (DataLoader): DataLoader for the specified mode
    """
    num_workers = data_cfg.num_workers
    batch_size = data_cfg.batch_size
    dataset_name = data_cfg.name

    dataset_class = find_dataset_using_name(dataset_name)

    sig = inspect.signature(dataset_class.__init__)

    # Check if 'cfg' is explicitly requested in the signature
    # Leave funcitonality for both styles of model class (those that take full cfg and those that take unpacked parameters)
    # Full cfg useful for development, experimentation and ablation stages, unpacked parameters for production and premade models
    if 'cfg' in sig.parameters:
        logging.info(f"Dataset [{dataset_name}] requested full 'cfg'. Passing OmegaConf object.")
        dataset = dataset_class(data_cfg, mode=mode)
    else:
        logging.info(f"Dataset [{dataset_name}] standardized. Unpacking parameters.")
        # Resolve and convert to standard dict
        params = OmegaConf.to_container(data_cfg, resolve=True)
        params.pop('name', None)
        dataset = dataset_class(**params, mode=mode)

    logging.info(f"Initialized {mode} dataset with {len(dataset)} samples.")

    sampler = DistributedSampler(dataset) if distributed else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Shuffle only if not using DistributedSampler
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    logging.info(f"Created DataLoader for {mode} with batch size {batch_size}, num_workers {num_workers}, distributed={distributed}.")

    return dataloader
 