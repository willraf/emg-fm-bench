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

def resolve_data_dir(cfg):
    """
    Resolve data directory based on TMPDIR environment variable.
    
    This is for the HPC as I copy data to a tmpdir for faster access."""
    tmpdir = os.environ.get("TMPDIR")
    data_dir = Path(cfg.data.data_dir)

    if tmpdir is not None:
        tmp_path = Path(tmpdir) / data_dir
        if tmp_path.exists():
            cfg.data.data_dir = str(tmp_path)
            return


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


def setup_data(cfg: OmegaConf, mode: str = "train", distributed: bool = False) -> DataLoader:
    """
    Setup data loaders and scheduler based on the configuration.
    
    Args:
        cfg (OmegaConf): Configuration object
        mode (str): One of ["train", "val", "test", "predict"]
        distributed (bool): Whether to use DistributedSampler for distributed training

    Returns:
        dataloader (DataLoader): DataLoader for the specified mode
    """
    num_workers = cfg.data.num_workers
    batch_size = cfg.data.batch_size

    # This is a bit of a hack, as I copy data to a tmpdir for faster access on the HPC
    # resolve_data_dir(cfg)

    dataset_class = find_dataset_using_name(cfg.data.name)
    dataset = dataset_class(cfg, mode=mode)

    sampler = DistributedSampler(dataset) if distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # Shuffle only if not using DistributedSampler
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    return dataloader
 