""" Module hdf5_windowed.py

TODO:
    Pad is signal is shorter than window length

Usage:


Authors:
    Will Raftery

Date:
    20/02/2026
"""
from dataclasses import dataclass, field

from pyparsing import Callable, Optional
from src.data.base import BaseDataset
import h5py
import torch
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

# @dataclass
# class HDF5WindowedDataset(BaseDataset):
#     cfg: DictConfig
#     mode: str = "train"  # or "val" or "test"
#     hdf5_path: Path
#     window_length: int
#     stride: Optional[int] = None
#     transforms: Optional[Callable] = None

#     # Define keys for metadata and data to be returned
#     # Metadata keys
#     user_metadata_keys: list[str] = field(default_factory=lambda: [
#         "subject_id"])
#     session_metadata_keys: list[str] = field(default_factory=lambda: [
#         "gesture_id"  ])
    
#     primary_data_key: str = "emg"
#     timeseries_keys: list[str] = field(default_factory=lambda: ["accel", "glove"])

#     def __post_init__(self):
#         self.stride = self.stride if self.stride is not None else self.window_length  # Default to non-overlapping windows
#         self.global_index = self._build_global_index()
#         self._file = None
#         transform_cfgs = self.cfg.data.transforms.get(self.mode)
#         self.transforms = self._build_transform(transform_cfgs) if transform_cfgs else None
    


class HDF5WindowedDataset(BaseDataset):

    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)

        self.mode = mode
        
        self.hdf5_path = Path(self.cfg.data.hdf5_path)
        self.window_length = self.cfg.data.window_length
        stride = self.cfg.data.stride
        self.stride = stride if stride is not None else self.window_length  # Default to non-overlapping windows

        self.primary_data_key = self.cfg.data.primary_data_key
        self.timeseries_keys = self.cfg.data.timeseries_keys if self.cfg.data.timeseries_keys is not None else []
        
        transform_cfgs = self.cfg.data.transforms.get(self.mode)
        self.transform = self._build_transform(transform_cfgs) if transform_cfgs else None

        # We build this ONCE at startup
        self.global_index = self._build_global_index()
        
        # Keep file handle None for now to be worker-safe
        self._file = None

    def _build_global_index(self) -> list[tuple[str, int]]:
        """
        Scan the master H5 file once and create a list of (internal_path, start_idx)
        """
        index = []
        # Open temporarily just to scan metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            # Recursively find all datasets that match our primary data key and check their session/user splits
            def scanner(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith(self.primary_data_key):
                    group_path = "/".join(name.split("/")[:-1])
                    session_group = f[group_path]
                    subject_group = session_group.parent
                    
                    # 1. Check Subject-Level Split
                    # For EPN612: 'train_user1' -> subject_split: 'train'
                    subj_split = subject_group.attrs.get('split', 'train')
                    
                    # 2. Check Session-Level Split (for EPN intra-user cases)
                    sess_split = session_group.attrs.get('split', 'train')

                    # LOGIC: 
                    # If the Subject is 'test', the whole subject belongs to the test set.
                    # If the Subject is 'train', we check if this specific session is 'test' (for val) or 'train' (for train).
                    # If the Subject is 'val', the whole subject belongs to the val set.
                    
                    is_match = False
                    if self.mode == "train":
                        if subj_split == "train" and sess_split == "train":
                            is_match = True
                    elif self.mode == "test":
                        if subj_split == "test" and sess_split == "train":
                            is_match = True
                    elif self.mode == "val":
                        if subj_split == "val":
                            is_match = True
                        if subj_split == "train" and sess_split == "test":
                            is_match = True

                    num_windows = 0
                    if is_match:
                        # Get length without loading data
                        T = obj.shape[0] 
                        # Calculate how many windows fit here
                        num_windows = (T - self.window_length) // self.stride + 1
                    
                    for w in range(num_windows):
                        start_time = w * self.stride
                        index.append((group_path, start_time))
            
            f.visititems(scanner)
        return index

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self):
        return len(self.global_index)

    def __getitem__(self, idx: int):
        # 1. Look up where this window lives
        group_path, start_idx = self.global_index[idx]

        session_group = self.file[group_path]
        user_group = session_group.parent
        end_idx = start_idx + self.window_length  

        # 2. Slice the data directly from the master file
        # HDF5 is very fast at random-access slicing
        primary_data = session_group[self.primary_data_key]
        primary_fs = primary_data.attrs.get("sample_rate")
        primary_window = primary_data[start_idx : end_idx]
        output = {self.primary_data_key: primary_window}

        # Go through other timeseries keys and slice them as well, applying scaling if sample rates differ
        for key in self.timeseries_keys:
            if key in session_group:
                data = session_group[key]
                sample_rate = data.attrs.get("sample_rate") 
                if sample_rate is not None:
                    output[key + "_sample_rate"] = sample_rate
                    scaling_factor = primary_fs / sample_rate
                    scaled_start_idx = int(start_idx * scaling_factor)
                    scaled_end_idx = int(end_idx * scaling_factor)
                    output[key] = data[scaled_start_idx : scaled_end_idx]
                else:
                    output[key] = data[start_idx : end_idx]

        # Just return all metadata for the session and user
        # Could specifiy which keys to return in the config if we want to be more selective
        for k, v in session_group.attrs.items():
            output[k] = v
        for k, v in session_group.parent.attrs.items():
            output[k] = v

        if self.transform:
            output = self.transform(output)
        return output