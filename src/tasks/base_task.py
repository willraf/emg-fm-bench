from torch.nn.parallel import DistributedDataParallel as DDP 

import torch.optim as optim
from torch.nn import Parameter

from omegaconf import OmegaConf
import logging

from typing import Iterator

from abc import ABC, abstractmethod


class BaseTask(ABC):

    # Abstract methods
    
    @abstractmethod
    def train_step(self, batch):
        pass
    
    @abstractmethod
    def val_step(self, batch):
        pass
    
    @abstractmethod
    def eval_step(self, batch):
        pass
    
    @abstractmethod
    def wrap_distributed(self, local_rank):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path):
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_path, epoch, **kwargs):
        pass
    
    @abstractmethod
    def modules(self):
        pass
    
    # Concrete methods

    def train(self):
        for m in self.modules():
            m.train()

    def eval(self):
        for m in self.modules():
            m.eval()
    
    # Static methods 

    @staticmethod
    def unwrap_model(model):
        """Utility function to unwrap a model from DDP if necessary.
        Prevents problems if loading a checkpoint saved from a DDP-wrapped model into a non-DDP model, or vice versa.
        """
        return model.module if isinstance(model, DDP) else model
    

