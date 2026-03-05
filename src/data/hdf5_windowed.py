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
import logging

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

        # The type of split/generalisation we are evaluating
        self.split_type = self.cfg.finetuning.split_type            # Should be either "user" or "session"
        assert self.split_type in ["user", "session"], "split_type must be either 'user' or 'session'"

        stride = self.cfg.data.stride
        self.stride = stride if stride is not None else self.window_length  # Default to non-overlapping windows
        self.jitter = self.cfg.data.jitter

        self.primary_data_key = self.cfg.data.primary_data_key
        self.root_key = self.cfg.data.root_key
        self.timeseries_keys = self.cfg.data.timeseries_keys if self.cfg.data.timeseries_keys is not None else []
        
        transform_cfgs = self.cfg.data.transforms.get(self.mode)
        self.transform = self._build_transform(transform_cfgs) if transform_cfgs else None

        # We build this ONCE at startup
        self.global_index = self._build_global_index()
        
        # Keep file handle None for now to be worker-safe
        self._file = None

        logging.info(f'{self.mode} dataset initialized with {len(self.global_index)} windows from {self.hdf5_path}')

    # def _build_global_index(self) -> list[tuple[str, int]]:
    #     """
    #     Scan the master H5 file once and create a list of (internal_path, start_idx)
    #     """
    #     index = []
    #     # Open temporarily just to scan metadata
    #     with h5py.File(self.hdf5_path, 'r') as f:
    #         # Recursively find all datasets that match our primary data key and check their session/user splits
    #         def scanner(name, obj):
    #             if isinstance(obj, h5py.Dataset) and name.endswith(self.primary_data_key):
    #                 group_path = "/".join(name.split("/")[:-1])
    #                 session_group = f[group_path]
    #                 subject_group = session_group.parent
                    
    #                 # 1. Check Subject-Level Split
    #                 # For EPN612: 'train_user1' -> subject_split: 'train'
    #                 subj_split = subject_group.attrs.get('split', 'train')
                    
    #                 # 2. Check Session-Level Split (for EPN intra-user cases)
    #                 sess_split = session_group.attrs.get('split', 'train')

    #                 # LOGIC: 
    #                 # If the Subject is 'test', the whole subject belongs to the test set.
    #                 # If the Subject is 'train', we check if this specific session is 'test' (for val) or 'train' (for train).
    #                 # If the Subject is 'val', the whole subject belongs to the val set.

    #                 is_match = False
    #                 if self.split_type == "user":
    #                     if self.mode == subj_split:
    #                         is_match = True
    #                 elif self.split_type == "session":
    #                     if self.mode == sess_split:
    #                         is_match = True

    #                 num_windows = 0
    #                 if is_match:
    #                     # Get length without loading data
    #                     T = obj.shape[0] 
    #                     # Calculate how many windows fit here
    #                     num_windows = (T - self.window_length) // self.stride + 1
                    
    #                 for w in range(num_windows):
    #                     start_time = w * self.stride
    #                     index.append((group_path, start_time))
            
    #         f.visititems(scanner)
    #     return index
    
    
    def _build_global_index(self):
        """
        Version using loops which may be faster
        """
        index = []
        i = 0
        with h5py.File(self.hdf5_path, 'r') as f:
            root = f[self.root_key] # e.g., "epn612"
            for user_name in root:
                user_group = root[user_name]
                subj_split = user_group.attrs.get('split', 'train')
                
                for sess_name in user_group:
                    session_group = user_group[sess_name]
                    sess_split = session_group.attrs.get('split', 'train')
                    
                    # Train/val/test split logic:
                    if sess_split == "unlabelled":
                        continue  # Skip unlabelled sessions for now, or we could add an "unlabelled" mode

                    is_match = False
                    if self.split_type == "user":
                        is_match = (self.mode == subj_split)
                        i += 1
                    else:
                        is_match = (self.mode == sess_split)
                    
                    if is_match:
                        obj = session_group[self.primary_data_key]
                        T = obj.shape[0]

                        # If the signal is shorter than the window length, we can still include it as one window (with padding applied in transforms)
                        if T < self.window_length:
                            num_windows = 1 
                        else:
                            num_windows = (T - self.window_length) // self.stride + 1

                        for w in range(num_windows):
                            index.append((f"{root.name}/{user_name}/{sess_name}", w * self.stride))
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

        # 2. Slice the data directly from the master file
        # HDF5 is very fast at random-access slicing
        primary_data = session_group[self.primary_data_key]
        T = primary_data.shape[0]

        if self.jitter and self.mode == "train":
            # Distance to the end of this sample
            leftover = T - (start_idx + self.window_length)

            # Handle case where signal is shorter than window length (leftover will be negative, so max_jitter will be 0 and no jitter applied)
            leftover = max(0, leftover)
            max_jitter = min(leftover, self.stride)  # Don't jitter more than the stride to avoid overlap with next window
            jitter_amount = np.random.randint(0, max_jitter + 1) if max_jitter > 0 else 0
            start_idx += jitter_amount

        end_idx = start_idx + self.window_length
        # Ensure we don't exceed trial boundaries
        actual_end_idx = min(end_idx, T)

        primary_fs = primary_data.attrs.get("sample_rate")
        primary_window = primary_data[start_idx : actual_end_idx]

        output = {self.primary_data_key: primary_window}

        # Go through other timeseries keys and slice them as well, applying scaling if sample rates differ
        for key in self.timeseries_keys:
            if key in session_group:
                data = session_group[key]
                sample_rate = data.attrs.get("sample_rate") 
                if sample_rate is not None:
                    output[key + "_sample_rate"] = sample_rate
                    # Scale the start and end indices so the time interval matches
                    scaling_factor = sample_rate / primary_fs
                    scaled_start_idx = round(start_idx * scaling_factor)
                    scaled_end_idx = scaled_start_idx + round(self.window_length * scaling_factor)
                    output[key] = data[scaled_start_idx : scaled_end_idx]
                else:
                    output[key] = data[start_idx : end_idx]

        # Just return all metadata for the session and user
        # Could specifiy which keys to return in the config if we want to be more selective
        for k, v in session_group.attrs.items():
            output[k] = v
        for k, v in user_group.attrs.items():
            output[k] = v

        for key, value in output.items():
            output[key] = self._auto_convert_type(key, value)

        if self.transform:
            output = self.transform(output)
        return output

    def _auto_convert_type(self, key: str, value):
        """
        Heuristic-based type conversion for a general-purpose dataset.
        """
        # Handle NumPy arrays (Signals, coordinates, etc.)
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            # If it's a floating point array, it's likely a signal -> Float32
            if torch.is_floating_point(tensor) or value.dtype == np.float16:
                return tensor.float()
            # If it's an integer array, it's likely labels or indices -> Long
            return tensor.long()

        # Handle Scalars (Metadata, Labels)
        if isinstance(value, (int, np.integer)):
            # Force everything that looks like an ID/Label to Long (int64)
            return torch.tensor(value, dtype=torch.long)
            
        if isinstance(value, (float, np.floating)):
            return torch.tensor(value, dtype=torch.float32)

        # Leave Strings/None/Booleans as they are (Dataloader will handle strings)
        return value