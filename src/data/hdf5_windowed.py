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
import sys

from pyparsing import Callable, Optional
from src.data.base import BaseDataset
import h5py
import torch
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
import logging


class HDF5WindowedDataset(BaseDataset):

    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)

        self.mode = mode
        
        self.hdf5_path = Path(cfg.hdf5_path)
        self.window_length = cfg.window_length

        # The type of split/generalisation we are evaluating
        self.split_type = cfg.split_type            # Should be either "user" or "session"
        assert self.split_type in ["inter_subject", "intra_session", "inter_session"], "split_type must be either 'inter_subject', 'intra_session', or 'inter_session'"

        stride = cfg.stride
        self.stride = stride if stride is not None else self.window_length  # Default to non-overlapping windows
        self.jitter = cfg.jitter

        self.primary_data_key = cfg.primary_data_key
        self.root_key = cfg.root_key
        self.timeseries_keys = cfg.timeseries_keys if cfg.timeseries_keys is not None else []
        
        transform_cfgs = cfg.transforms.get(self.mode)
        self.transform = self._build_transform(transform_cfgs) if transform_cfgs else None

        # We build this ONCE at startup
        self.global_index = self._build_global_index()

        # Log stats about memery size of index
        if len(self.global_index) > 0:
            example_path, example_start_idx = self.global_index[0]
            logging.info(f"Memory size of global index: {sys.getsizeof(example_path) + sys.getsizeof(example_start_idx) * len(self.global_index)} bytes")

        # Keep file handle None for now to be worker-safe
        self._file = None

        logging.info(f'{self.mode} dataset initialized with {len(self.global_index)} windows from {self.hdf5_path}')
    
    
    def _build_global_index(self):
        """
        Version using loops which may be faster
        """
        index = []
        with h5py.File(self.hdf5_path, 'r') as f:
            root = f[self.root_key] # e.g., "epn612"
            for user_key in root:
                user_group = root[user_key]
                
                for session_key in user_group:
                    session_group = user_group[session_key]

                    for run_key in session_group:
                        run_group = session_group[run_key]

                        is_match = False
                        if self.split_type == "inter_subject":
                            if "inter_subject_split" in run_group.attrs:
                                subject_split = run_group.attrs['inter_subject_split']
                            elif 'inter_subject_split' in session_group.attrs:
                                subject_split = session_group.attrs['inter_subject_split']
                            elif 'inter_subject_split' in user_group.attrs:
                                subject_split = user_group.attrs['inter_subject_split']
                            else:
                                raise ValueError(f"Missing 'inter_subject_split' attribute in both session and user groups for {user_key}/{session_key}/{run_key}. Please ensure your HDF5 file has the correct metadata structure.")
                            is_match = (self.mode == subject_split)

                        elif self.split_type == "inter_session":
                            if 'inter_session_split' in run_group.attrs:
                                inter_session_split = run_group.attrs['inter_session_split']
                            elif "inter_session_split" in session_group.attrs:
                                inter_session_split = session_group.attrs['inter_session_split']
                            else:
                                raise ValueError(f"Missing 'inter_session_split' attribute in session group for {user_key}/{session_key}/{run_key}. Please ensure your HDF5 file has the correct metadata structure.")
                            is_match = (self.mode == inter_session_split)


                        elif self.split_type == "intra_session":
                            if 'intra_session_split' in run_group.attrs:
                                intra_session_split = run_group.attrs['intra_session_split']
                            else:
                                raise ValueError(f"Missing 'intra_session_split' attribute in session group for {user_key}/{session_key}/{run_key}. Please ensure your HDF5 file has the correct metadata structure.")
                            is_match = (self.mode == intra_session_split)

                        else:
                            raise ValueError(f"Invalid split_type: {self.split_type}. Must be one of 'inter_subject', 'intra_session', or 'inter_session'.")

                        
                        if is_match:
                            obj = run_group[self.primary_data_key]
                            T = obj.shape[0]

                            # If the signal is shorter than the window length, we can still include it as one window (with padding applied in transforms)
                            if T < self.window_length:
                                num_windows = 1 
                            else:
                                num_windows = (T - self.window_length) // self.stride + 1

                            for w in range(num_windows):
                                index.append((f"{root.name}/{user_key}/{session_key}/{run_key}", w * self.stride))
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
        raw_slice = primary_data[start_idx : actual_end_idx]

        # Ensure output window is always the same length, even if the whole signal is shorter than window length
        primary_window = self.pad_crop_to_length(raw_slice, self.window_length)

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
                    scaled_window_length = round(self.window_length * scaling_factor)
                    
                    scaled_end_idx = scaled_start_idx + scaled_window_length
                    # Pad or crop to ensure consistent length
                    # Should only happen is the full singal is shorter than the window length after scaling, in which case we pad from the left
                    window = self.pad_crop_to_length(data[scaled_start_idx : scaled_end_idx], scaled_window_length)
                else:
                    window = self.pad_crop_to_length(data[start_idx : end_idx], self.window_length)

                output[key] = window

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
    
    @staticmethod
    def pad_crop_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad with zeros if signal is shorter than target_length, or crop if it's longer.
        Same logic as PadCropToLength transform

        Args:
            signal: np.ndarray of shape (T, C)
            target_length: int, desired length of the output signal
        """
        current_length = signal.shape[0]
        if current_length < target_length:
            pad_width = target_length - current_length
            # Pad the time dimension (0) with zeros on the left side
            return np.pad(signal, ((pad_width, 0), (0, 0)), mode='constant')
        else:
            return signal[:target_length]