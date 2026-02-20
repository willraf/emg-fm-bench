"""
This is an experimental file for testing out a new approach to loading the EPN dataset (and other large datasets in the future)

The idea it to store the whole dataset (or each split) in a single HDF5 file, with the following structure:

- ninapro_db5 (root group)
    - S1
        - .attrs (user metadata) ['age', 'circumference_cm', 'gender', 'height_cm', 'laterality', 'weight_kg']>
        - E1_R1_G1 (exercise 1, repetition 1, gesture 1)
            - .attrs (session metadata) ['gesture_id', 'repetition_id']
            - emg (dataset of shape (T, 16))
            - accel (dataset of shape (T, 3))
            - glove (dataset of shape (T, 22))
        - E1_R1_G2
        - ...
    - S2

This allows us to load the whole dataset for a split with a single file handle.

to handle indexing of each window within each session, we can either:
1. Store a separate index file that maps from global window index to (user, session, start_time), or even just stores the length of each session
2. On creation of the dataset, precompute the index mapping of each window and store it
"""

from src.data.base import BaseDataset
import h5py
import torch
from pathlib import Path

class NinaProDB5Dataset(BaseDataset):
    def __init__(self, cfg, mode="train"):
        super().__init__(cfg)

        data_dir = Path(self.cfg.data.data_dir)
        self.window_length = self.cfg.data.window_length
        stride = self.cfg.data.stride
        self.stride = stride if stride is not None else self.window_length  # Default to non-overlapping windows

        if mode == "train":
            Warning("The current NinaPro DB5 is used only for validation/testing")
        
        self.h5_path = data_dir / f"ninapro_db5.hdf5"

        transform_cfgs = cfg.data.transforms.get(mode)
        self.window_transform, self.sample_transform = self._build_transform(transform_cfgs) if transform_cfgs else None

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

        # 3. Kinematic data
        # accel_window = self.file[parent_path]['accel'][start_idx : start_idx + self.window_length]
        # glove_window = self.file[parent_path]['glove'][start_idx : start_idx + self.window_length]
        
        # 4. Get metadata from the parent group of the emg dataset
        gesture_id = self.file[parent_path].attrs["gesture_id"]

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