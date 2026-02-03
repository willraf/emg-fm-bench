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
from omegaconf import OmegaConf, DictConfig
from typing import Sequence, Any
from hydra.utils import instantiate
import pandas as pd
from pathlib import Path

from src.data import transforms
from src.data.transforms import Transform

from src.data.emg2pose_data import WindowedEmgDataset, SingleChannelEmgDataset

from torch.utils.data import ConcatDataset, DataLoader


from torch.utils.data.distributed import DistributedSampler


import logging
import os

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


def load_splits(
    cfg: OmegaConf,
) -> dict[str, list[str]]:
    """Load train, val, and test datasets from metadata csv.
    
    Returns:
        splits (dict): Dictionary with keys "train", "val", "test" and values as lists of filenames.
    """
    metadata_file = cfg.data.metadata_file
    subsample = cfg.data.subsample
    random_seed = cfg.experiment.random_seed
    data_dir = cfg.data.data_dir

    # Load dataframe
    df = pd.read_csv(metadata_file)

    # Optionally subsample
    df = df.groupby("split").apply(
        lambda x: x.sample(frac=subsample, random_state=random_seed),
    )
    df.reset_index(drop=True, inplace=True)

    base_dir = Path(data_dir).expanduser()

    # Format as dictionary
    splits = {}
    for split, df_ in df.groupby("split"):
        splits[split] = [
            str(base_dir.joinpath(f'{filename}.hdf5')) 
            for filename in df_.filename if base_dir.joinpath(f'{filename}.hdf5').is_file()
        ]

    return splits

# Instantiate transforms
def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
    return transforms.Compose([instantiate(cfg) for cfg in configs])

def setup_data(cfg: OmegaConf, mode: str = "train"):
    """
    Setup data loaders and scheduler based on the configuration.
    
    Args:
        cfg (OmegaConf): Configuration object
        mode (str): One of ["train", "val", "test", "predict"]

    Returns:
        dataloaders (dict): Dict of DataLoaders (train/val/test) as appropriate
    """
    num_workers = cfg.data.num_workers
    window_length = cfg.data.window_length
    skip_ik_failures = cfg.data.skip_ik_failures
    batch_size = cfg.data.batch_size
    
    num_channels = cfg.data.num_channels

    transform_cfgs = cfg.data.transforms.get(mode)

    resolve_data_dir(cfg)

    if mode == "train":
        jitter = True
        transform_cfgs = cfg.data.transforms.train
        padding = cfg.data.padding
    else:
        padding = (0, 0)
        jitter = False

    if transform_cfgs is not None:
        transform = _build_transform(transform_cfgs)
    else:
        transform = None
    
    splits = load_splits(cfg)
    sessions = splits[mode]
    
    if num_channels == 1:
        # For separate channels
        dataset = ConcatDataset(
                [
                    SingleChannelEmgDataset(
                        hdf5_path,
                        transform=transform,
                        window_length=window_length,
                        padding=padding,
                        jitter=jitter,
                        skip_ik_failures=skip_ik_failures,
                    )
                    for hdf5_path in sessions
                ]
            )
        
    else:
        # For training on all channels
        dataset = ConcatDataset(
                [
                    WindowedEmgDataset(
                        hdf5_path,
                        transform=transform,
                        window_length=window_length,
                        padding=padding,
                        jitter=jitter,
                        skip_ik_failures=skip_ik_failures,
                    )
                    for hdf5_path in sessions
                ]
            )
        
    sampler = DistributedSampler(dataset) if mode == "train" else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True, 
    # )

    return dataloader