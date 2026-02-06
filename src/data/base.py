""" Module base_dataset

This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

Usage:
    class MyDataset(BaseDataset):
        def __init__(self, cfg):
            super().__init__(cfg)
            # Initialize your dataset here

Authors:
    Will Raftery

Date:
    31/07/2025
"""
from abc import ABC, abstractmethod

from omegaconf import OmegaConf

from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__ method.")

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__ method.")

    def pre_epoch_callback(self, epoch):
        """Callback to be called before every epoch.
        """
        pass

    def post_epoch_callback(self, epoch):
        pass


    def update_parameters(self, **kwargs) -> None:
        """Update the parameters of the generator.

        Useful for curriculum learning
        NOTE ** is kind of pointless here, as we always pass a dict, but it keeps usage general

        Args:
            kwargs (Optional[dict]): Dictionary of parameters to update.
        """
        if kwargs is None:
            return
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Dataset does not have attribute '{key}' to update.")