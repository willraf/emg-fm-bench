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

from src.utils.utils import load_config, get_experiment_directory

from scripts.train import main as train
from scripts.eval import main as evaluate
from scripts.predict import main as predict

import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    experiment_dir = get_experiment_directory(cfg)
    OmegaConf.save(cfg, experiment_dir / "config.yaml")

    train_results = train(cfg, overwrite=True)
    logging.info(f"Training completed with results: {train_results}")

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
    main()
