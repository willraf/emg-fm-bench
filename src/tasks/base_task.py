from torch.nn.parallel import DistributedDataParallel as DDP 

import torch.optim as optim
from torch.nn import Parameter

from omegaconf import OmegaConf
import logging

from typing import Iterator

class BaseTask:
    def train_step(self, batch):
        raise NotImplementedError
    
    def val_step(self, batch):
        raise NotImplementedError
    
    def eval_step(self, batch):
        raise NotImplementedError
    
    def wrap_distributed(self, local_rank):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_path, epoch, **kwargs):
        raise NotImplementedError
    

    def train(self):
        for m in self.modules():
            m.train()

    def eval(self):
        for m in self.modules():
            m.eval()

    def modules(self):
        return []
    
    
    def setup_optimiser(self, model_parameters: Iterator[Parameter]) -> optim.Optimizer:
        """
        Create optimiser from config file
        """
        optimizer_name = self.cfg.training.optimizer
        learning_rate = self.cfg.training.learning_rate
        momentum = self.cfg.training.momentum
        weight_decay = self.cfg.training.weight_decay

        if optimizer_name == 'sgd':
            optimizer = optim.SGD(
                model_parameters, 
                lr=learning_rate, 
                momentum=momentum
            )
            logging.info("Using SGD optimizer")
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                model_parameters, 
                lr=learning_rate,
                weight_decay=weight_decay
            )
            logging.info("Using Adam optimizer")
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model_parameters, 
                lr=learning_rate,
                weight_decay=weight_decay
            )
            logging.info("Using Adam optimizer")
        else:
            raise ValueError("Optimizer not recognised")

        return optimizer

    
    @staticmethod
    def unwrap_model(model):
        """Utility function to unwrap a model from DDP if necessary.
        Prevents problems if loading a checkpoint saved from a DDP-wrapped model into a non-DDP model, or vice versa.
        """
        return model.module if isinstance(model, DDP) else model
    

