""" 
Script main

Main entry point for the full training/evaluation/prediction pipeline.

Usage:
    % python -m scripts.main --config configs/main.yaml
    % python -m scripts.main --config configs/main.yaml --default configs/default.yaml
    % python -m torch.distributed.run --nproc_per_node=1 -m scripts.main --config configs/main.yaml
    
    With Hydra:
    % python -m scripts.main --config-name other_config
    % python -m torch.distributed.run   --nproc_per_node=4 -m scripts.main --config-name other_config

Authors:
    Will Raftery

Date:
    31/07/2025
"""

import argparse
import logging

from omegaconf import OmegaConf

from src.utils.utils import load_config, get_experiment_directory, setup_logging

from scripts.train import main as train
from scripts.eval import main as evaluate
from scripts.predict import main as predict

import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):

    find_batch_size = cfg.experiment.get('find_batch_size', False)

    experiment_dir = get_experiment_directory(cfg)

    if find_batch_size:
        batch_size = train(cfg, overwrite=True, find_batch_size=True)
        logging.info(f"Found maximum batch size: {batch_size}")
        cfg.data.batch_size = batch_size

        # Adjust learning rate based on batch size (linear scaling rule)
        base_batch_size = cfg.experiment.get('base_batch_size', 128)
        base_learning_rate = cfg.experiment.get('base_learning_rate', 1e-3)
        scaled_learning_rate = base_learning_rate * (batch_size / base_batch_size)
        cfg.finetuning.learning_rate = scaled_learning_rate
        logging.info(f"Scaled learning rate to {scaled_learning_rate} based on batch size {batch_size} and base batch size {base_batch_size}.")

    OmegaConf.save(cfg, experiment_dir / "config.yaml")

    train_results = train(cfg, overwrite=True, find_batch_size=cfg.experiment.find_batch_size)
    logging.info(f"Training completed with results: {train_results}")

    if find_batch_size:
        # Optimization: Eval can usually handle 2x the training batch size
        cfg.data.batch_size = batch_size * 2
        logging.info(f"Set evaluation batch size to {cfg.data.batch_size} based on discovered training batch size.")

    eval_results = evaluate(cfg, overwrite=True)
    logging.info(f"Evaluation completed with results: {eval_results}")

    # infer_results = predict(cfg, overwrite=overwrite)
    # logging.info(f"Inference completed with results: {infer_results}")

    return {
        "train_results": train_results,
        "eval_results": eval_results,
        # "infer_results": infer_results
    }

if __name__ == "__main__":
    setup_logging()
    main()
