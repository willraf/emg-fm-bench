"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel
"""

import importlib
from torch.optim import lr_scheduler
from omegaconf import OmegaConf 

import logging

from src.tasks.base_task import BaseTask


def find_task_using_name(task_name: str):
    """
    Import the module "src/tasks/[task_name].py".
    The class [TaskName] must exist inside and inherit from BaseModel.
    """
    module_name = f"src.tasks.{task_name}_task"
    try:
        modellib = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import module {module_name}") from e

    target_class_name = task_name.replace('_', '') + 'task'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_class_name.lower():
            return cls
        
    raise ImportError(f"Expected class '{target_class_name}' in '{module_name}'.")


def setup_task(cfg: OmegaConf, device: str, distributed=False, is_main=False, save_dir=None) -> BaseTask:
    """Create a model given the configuration.

    This is the main interface between this package and train.py/validate.py
    """
    task_name = cfg.task.name
    task = find_task_using_name(task_name)
    instance = task(cfg, device=device, distributed=distributed, is_main=is_main, save_dir=save_dir)
    logging.info("Task [{0}] was created".format(type(instance).__name__))
    return instance