""" Module epn_612.py

Each HDF5 file contains the following structure
- epn612 (root group)
    - .attrs: name, age, gender, occupation, ethnicGroup, handedness, ArmDamage, distanceFromElboeToMyoInCm, 
                distanceFromElbowToUlnaInCm, armPerimeterInCm, date, deviceModel, samplingFrequencyInHertz (of EMG), recordingTimeInSeconds,
                repetitionsForSynchronizationGesture, gesture_name, start_point, session
    - emg (dataset): shape (8, T), dtype float32
    - imu (group)
        - accelerometer (dataset): shape (3, T), dtype float32
        - gyroscope (dataset): shape (3, T), dtype float32
        - quaternion (dataset): shape (4, T), dtype float32


TODO:  
    - Add jitter to window offsets for training time variability
    - Add padding
    - Get only part of the signal containing the gesture

Usage:


Authors:
    Will Raftery

Date:
    16/02/2026
"""


from __future__ import annotations

from collections.abc import KeysView

from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from src.data import transforms
from src.data.transforms import Transform
from src.data.base import BaseDataset

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from typing import Sequence

import pandas as pd




class EPN612Dataset(BaseDataset):

    def __init__(self, cfg: OmegaConf, mode: str):
        super().__init__(cfg)
        self.cfg = cfg
        self.mode = mode

        self.dataset = self._build_dataset()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


    def _build_dataset(self):
        cfg = self.cfg
        mode = self.mode   
        
        window_length = cfg.data.window_length
        jitter = (mode == "train")
        padding = cfg.data.padding if mode == "train" else (0, 0)

        transform_cfgs = cfg.data.transforms.get(mode)
        window_transform, sample_transform = self._build_transform(transform_cfgs) if transform_cfgs else None

        if mode == "train":
            dataset_dir = Path(cfg.data.data_dir) / "trainingHDF5" / "training"
        elif mode == "val":
            dataset_dir = Path(cfg.data.data_dir) / "trainingHDF5" / "testing"
        elif mode == "test":
            dataset_dir = Path(cfg.data.data_dir) / "testingHDF5" / "testing"

        session_files = sorted(dataset_dir.glob("*.hdf5"))

        datasets = [
            WindowedEmgDataset(
                hdf5_path,
                window_transform=window_transform,
                sample_transform=sample_transform,
                window_length=window_length,
                padding=padding,
                jitter=jitter,
            )
            for hdf5_path in session_files
        ]
        
        return ConcatDataset(datasets)




@dataclass
class EPN612SessionData:
    """A read-only interface to a single epn612 session file stored in
    HDF5 format.

    ``self.imu`` is a `h5py.Dataset` instance with a compound data type
    as in a numpy structured array containing three fields - accelerometer, gyroscope and quaternion.
    The sampling rate of EMG is 200Hz, each EMG device has 8 electrode
    channels.

    NOTE: Only the metadata and ground-truth are loaded into memory while the
    EMG data is accesssed directly from disk. When wrapping this interface
    within a PyTorch Dataset, use multiple dataloading workers to mask the
    disk seek and read latencies."""

    HDF5_GROUP: ClassVar[str] = "epn612"
    # timeseries keys
    IMU: ClassVar[str] = "imu"
    EMG: ClassVar[str] = "emg"
    ACCELEROMETER: ClassVar[str] = "accelerometer"
    GYROSCOPE: ClassVar[str] = "gyroscope"
    QUATERNION: ClassVar[str] = "quaternion"
    # metadata keys
    GESTURE_NAME: ClassVar[str] = "gesture_name"
    START_POINT: ClassVar[str] = "start_point"
    SESSION_ID: ClassVar[str] = "session"

    USER_ID: ClassVar[str] = "name"
    AGE: ClassVar[str] = "age"
    GENDER: ClassVar[str] = "gender"
    ETHINIC_GROUP: ClassVar[str] = "ethnicGroup"
    SAMPLE_RATE: ClassVar[str] = "samplingFrequencyInHertz"

    OCCUPATION: ClassVar[str] = "occupation"
    HANDEDNESS: ClassVar[str] = "handedness"
    ARM_DAMAGE: ClassVar[str] = "ArmDamage"
    DISTANCE_FROM_ELBOE_TO_MYO_IN_CM: ClassVar[str] = "distanceFromElboeToMyoInCm"
    DISTANCE_FROM_ELBOW_TO_ULNA_IN_CM: ClassVar[str] = "distanceFromElbowToUlnaInCm"   
    ARM_PERIMETER_IN_CM: ClassVar[str] = "armPerimeterInCm"
    DATE: ClassVar[str] = "date"
    DEVICE_MODEL: ClassVar[str] = "deviceModel"
    RECORDING_TIME_IN_SECONDS: ClassVar[str] = "recordingTimeInSeconds"
    REPETITIONS_FOR_SYNCHRONIZATION_GESTURE: ClassVar[str] = "repetitionsForSynchronizationGesture"

    hdf5_path: Path


    gesture_mapping: ClassVar[dict[str, int]] = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}


    def __post_init__(self) -> None:
        # Just load the metadata and length
        with h5py.File(self.hdf5_path, "r") as f:
            epn612_group: h5py.Group = f[self.HDF5_GROUP]
            self.length = len(epn612_group[self.EMG]) # Assumes (T, C) shape from conversion script

            # Load the metadata entirely into memory
            self.metadata: dict[str, Any] = {}
            for key, val in epn612_group.attrs.items():
                self.metadata[key] = val

        self._file = None

    # Lazy load the EMG and IMU datasets when first accessed
    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file


    def __enter__(self) -> EPN612SessionData:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if hasattr(self, "_file"):
            self._file.close()

    def __len__(self) -> int:
        return self.length
    

    def __getitem__(self, key: slice) -> np.ndarray:
        # return self.emg[:, key]
    
        if isinstance(key, int):
            key = slice(key, key + 1)
        
        # 1. Fetch EMG (200Hz)
        emg = self.file[self.HDF5_GROUP][self.EMG]  # h5py.Dataset
        emg_slice = emg[key]

        imu = self.file[self.HDF5_GROUP][self.IMU]  # h5py.Dataset with compound dtype

        # 2. Fetch IMU (50Hz) - Scale the slice by 1/4
        # Note: This is an approximation. For exact windows, 
        # use start = key.start // 4, stop = key.stop // 4
        imu_start = key.start // 4 if key.start is not None else None
        imu_stop = key.stop // 4 if key.stop is not None else None
        imu_slice = slice(imu_start, imu_stop)

        return {
            "emg": emg_slice,
            # "accel": imu[self.ACCELEROMETER][:, imu_slice],
            # "gyro": imu[self.GYROSCOPE][:, imu_slice],
            # "quat": imu[self.QUATERNION][:, imu_slice]
        }
    

    @property
    def gesture_name(self) -> str:
        return self.metadata.get(self.GESTURE_NAME, 'unknown')
    
    @property
    def gesture_id(self) -> int:
        return self.gesture_mapping.get(self.gesture_name, -1)

    @property
    def start_point(self) -> int:
        return self.metadata.get('start_point', 0)
    
    @property
    def session_name(self) -> str:
        return f"{self.metadata.get(self.USER_ID)}_{self.metadata.get(self.SESSION_ID, 'unknown')}"
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.session_name} ({len(self)} samples)"


@dataclass
class WindowedEmgDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` corresponding to an instance of
    `Emg2PoseSessionData` that iterates over EMG windows of configurable
    length and stride.

    Args:
        hdf5_path (str): Path to the session file in hdf5 format.
        window_length (int): Size of each window. Specify None for no windowing
            in which case this will be a dataset of length 1 containing the
            entire session. (default: ``None``)
        stride (int): Stride between consecutive windows. Specify None to set
            this to window_length, in which case there will be no overlap
            between consecutive windows. (default: ``window_length``)
        padding (tuple[int, int]): Left and right contextual padding for
            windows in terms of number of raw EMG samples.
        jitter (bool): If True, randomly jitter the offset of each window.
            Use this for training time variability. (default: ``False``)
        window_transform (Callable): A composed sequence of transforms that takes
            a window/slice of `Emg2PoseSessionData` in the form of a numpy
            structured array and returns a `torch.Tensor` instance.
            (default: ``emg2pose.transforms.ExtractToTensor()``)
        sample_transform (Callable): A composed sequence of transforms that takes
            in the entire sample dict
    """

    hdf5_path: Path
    window_length: InitVar[int | None] = 10_000
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    # Operates on individual windows of emg
    window_transform: Transform[np.ndarray, torch.Tensor] = field(
        default_factory=transforms.ExtractToTensor
    )
    # Operates on the entire sample dict (emg, joint angles, no_ik_failure mask) after windowing and before returning
    sample_transform: Transform[dict, dict] = None

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
        self.session_length = len(self.session)
        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        # Precompute (start, end) windows based on skip_ik_failures setting
        self.windows: list[tuple[int, int]] = self.precompute_windows()

    def __len__(self) -> int:
        return (len(self.session) - self.window_length) // self.stride + 1


    @property
    def session(self):  
        if not hasattr(self, "_session"):
            self._session = EPN612SessionData(self.hdf5_path)
        return self._session


    def precompute_windows(self) -> list[tuple[int, int]]:
        """For each dataset idx, precompute the EMG start and times."""
        windows = []

        for idx in range(len(self)):
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_length
            if end_idx > self.session_length:
                break
            windows.append((start_idx, end_idx))
        
        return windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        start_idx, end_idx = self.windows[idx]
        window = self.session[start_idx:end_idx]

        emg = self.window_transform(window)
        assert torch.is_tensor(emg)

        sample = {
            "emg": emg.T,  # CT
            "window_start_idx": start_idx,
            "window_end_idx": end_idx,
            "gesture_id": self.session.gesture_id,
        }
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return sample
    
        start_idx, end_idx = self.windows[idx]
        leftover

        # Randomly jitter the window offset
        offset, end_idx = self.windows[idx]
        leftover = end_idx - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds, leftover {leftover}")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding and fetch
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]

        # Extract EMG tensor corresponding to the window
        emg = self.window_transform(window)
        assert torch.is_tensor(emg)

        # Extract joint angle labels
        joint_angles = window[Emg2PoseSessionData.JOINT_ANGLES]
        joint_angles = torch.as_tensor(joint_angles)

        # Mask of samples without IK failures
        no_ik_failure = torch.as_tensor(
            self.session.no_ik_failure[window_start:window_end]
        )

        # Convert stage to index

        sample = {
            "emg": emg.T,  # CT
            "joint_angles": joint_angles.T,  # CT
            "no_ik_failure": no_ik_failure,  # T
            "window_start_idx": window_start,
            "window_end_idx": window_end,
            "stage": self._stage,
        }

        if self.sample_transform is not None:
            sample = self.sample_transform(sample) 
        return sample
    




