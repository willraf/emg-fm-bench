"""
This is an experimental file for testing out a new approach to loading the EPN dataset (and other large datasets in the future)

The idea it to store the whole dataset (or each split) in a single HDF5 file, with the following structure:

- epn_612
    - user1
        - .attrs (user metadata)
        - session1
            - .attrs (session metadata)
            - emg (dataset of shape (T, C))
            - imu (dataset of shape (T, 10) with compound dtype for accel, gyro, quat)
        - session2
    - user2

This allows us to load the whole dataset for a split with a single file handle.

to handle indexing of each window within each session, we can either:
1. Store a separate index file that maps from global window index to (user, session, start_time), or even just stores the length of each session
2. On creation of the dataset, precompute the index mapping of each window and store it
"""

from src.data.base import BaseDataset
import h5py
import torch
from pathlib import Path


class EPN612Dataset(BaseDataset):
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)

        data_dir = Path(self.cfg.data.data_dir)
        self.window_length = self.cfg.data.window_length
        stride = self.cfg.data.stride
        self.stride = stride if stride is not None else self.window_length  # Default to non-overlapping windows

        if mode == "train":
            self.h5_path = data_dir / "training" / "training.hdf5"
        elif mode == "val":
            self.h5_path = data_dir / "training" / "testing.hdf5"
        elif mode == "test":
            self.h5_path = data_dir / "testing" / "testing.hdf5"

        transform_cfgs = cfg.data.transforms.get(mode)
        self.window_transform, self.sample_transform = self._build_transform(transform_cfgs) if transform_cfgs else None

        # We build this ONCE at startup
        self.global_index = self._build_global_index()
        
        # Keep file handle None for now to be worker-safe
        self._file = None

        self.gesture_to_id = {'noGesture': 0, 'fist': 1, 'waveIn': 2, 'waveOut': 3, 'open': 4, 'pinch': 5}

    def _build_global_index(self) -> list[tuple[str, int]]:
        """
        Scan the master H5 file once and create a list of (internal_path, start_idx)
        """
        index = []
        # Open temporarily just to scan metadata
        with h5py.File(self.h5_path, 'r') as f:
            # Recursively find all 'emg' datasets
            def scanner(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("emg"):
                    # Get length without loading data
                    total_len = obj.shape[0] 
                    # Calculate how many windows fit here

                    num_windows = (total_len - self.window_length) // self.stride + 1
                    
                    for w in range(num_windows):
                        start_time = w * self.stride
                        index.append((name, start_time))
            
            f.visititems(scanner)
        return index

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.global_index)

    def __getitem__(self, idx: int):
        # 1. Look up where this window lives
        internal_path, start_idx = self.global_index[idx]
        parent_path = "/".join(internal_path.split("/")[:-1])
        
        # 2. Slice the data directly from the master file
        # HDF5 is very fast at random-access slicing
        emg_window = self.file[internal_path][start_idx : start_idx + self.window_length]

        # 3. Fetch IMU (50Hz) - Scale the slice by 1/4
        # Note: This is an approximation. For exact windows, 
        # use start = key.start // 4, stop = key.stop // 4
        imu_start = start_idx // 4  # Assuming IMU is at 50Hz and EMG is at 200Hz
        imu_stop = (start_idx + self.window_length) // 4
        imu_window = self.file[parent_path]['imu'][imu_start : imu_stop]  # h5py.Dataset with compound dtype
        
        # 4. Get metadata from the parent group of the emg dataset
        gesture_name = self.file[parent_path].attrs["gesture_name"]

        gesture_id = self.gesture_to_id.get(gesture_name, -1)  # Map to ID, default to -1 if not found

        window = {"emg": emg_window}
        if self.window_transform is not None:
            emg = self.window_transform(window)
        assert torch.is_tensor(emg)

        sample = {
            "emg": emg.T, # (C, T)
            "gesture_id": gesture_id,
            "path": internal_path,
            "window_start_idx": start_idx,
            "window_end_idx": start_idx + self.window_length,
        }
        if self.sample_transform:
            sample = self.sample_transform(sample)
        return sample