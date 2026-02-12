"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel
"""

import importlib
from torch.optim import lr_scheduler
from omegaconf import OmegaConf 
from typing import Optional
import logging

from src.models.soundstream.discriminator import STFTDiscriminator, WaveDiscriminator


def find_model_using_name(model_name: str):
    """
    Import the module "src/models/[model_name].py".
    The class [ModelName] must exist inside and inherit from BaseModel.
    """
    module_name = f"src.models.{model_name}"
    try:
        modellib = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import module {module_name}") from e
    
    target_class_name = model_name.replace('_', '')
    # target_class_name = f"{model_name.capitalize()}"

    for name in dir(modellib):
        obj = getattr(modellib, name)
        print(name.lower(), target_class_name.lower(), isinstance(obj, type))
        if name.lower() == target_class_name.lower() and isinstance(obj, type): # issubclass(obj, BaseDataset):
            return obj

    raise ImportError(f"Expected class '{target_class_name}' in '{module_name}'.")


def setup_model(cfg: OmegaConf, device: str, model_name: Optional[str] = None):
    """Create a model given the configuration.

    This is the main interface between this package and train.py/validate.py
    """
    if model_name is None:
        model_name = cfg.model.name
        
    model = find_model_using_name(model_name)
    instance = model(cfg)
    instance.to(device)
    logging.info("model [{0}] was created".format(type(instance).__name__))
    return instance

def setup_discriminator(cfg: OmegaConf, device: str):
    """Create a discriminator given the configuration.

    This is the main interface between this package and train.py/validate.py
    """
    in_channels = cfg.data.num_channels
    C = cfg.discriminator.stft.channel_multiplier
    window_length = cfg.discriminator.stft.window_length
    f_bins = window_length // 2 + 1
    stft_disc = STFTDiscriminator(in_channels=2, C=C, frequency_bins=f_bins).to(device)

    num_discs = cfg.discriminator.wave.num_discs
    first_dim = cfg.discriminator.wave.first_dim
    n_layers = cfg.discriminator.wave.n_layers
    downsampling_factor = cfg.discriminator.wave.downsampling_factor
    wave_disc = WaveDiscriminator(in_channels=in_channels, num_D=num_discs, ndf=first_dim, n_layers=n_layers, downsampling_factor=downsampling_factor).to(device)

    return stft_disc, wave_disc