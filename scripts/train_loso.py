""" Script train_loso.py

Leave-One-Out cross-validation wrapper around the standard training pipeline.
Iterates over all subjects in the HDF5 file, designating each as the test
subject in turn, and aggregates results across folds.

Usage:
    $ python -m scripts.train_loso --config_name configs/ninapro_ablation.yaml
    $ python -m scripts.train_loso +experiment=full_sweep

Date:
    17/05/2026
"""

import logging
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf, DictConfig

from src.utils.utils import load_config, set_seed, get_experiment_directory, setup_logging, setup_save_dir
from scripts.main import main as train_main
import hydra


def get_subject_ids(hdf5_path: str, root_key: str) -> list[str]:
    """Read all subject IDs from the HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        subj_keys = f[root_key].keys()
        subj_keys = [k for k in subj_keys if f[root_key][k].attrs.get('inter_subject_split') != 'test']
        return sorted(subj_keys)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    """
    Main provides an entry point for the full pipeline.
    It handles:
    - Set random seed
    - Distributed
    - Logging & wandb
    - Setting up directory structures and saving
    - Trainer init, batch size discovery, and running
    - Evaluator init, batch size discovery, and running
    - Tear down
    """

    experiment_name = cfg.experiment.name
    if cfg.data.split_type != 'loso':
        raise ValueError(f"Expected split_type 'loso' for LOSO training, but got '{cfg.data.split_type}'")
    
    # ------------------------------------------------------------------
    # Discover subjects
    # ------------------------------------------------------------------
    subject_ids = get_subject_ids(cfg.data.hdf5_path, cfg.data.root_key)
    logging.info(f"LOSO cross-validation over {len(subject_ids)} subjects: {subject_ids}")

    fold_results = []

    for fold_idx, test_subject in enumerate(subject_ids):
        logging.info(f"\n{'='*60}")
        logging.info(f"Fold {fold_idx + 1}/{len(subject_ids)} — test subject: {test_subject}")
        logging.info(f"{'='*60}")

        # Clone config and inject LOSO subject
        fold_cfg = OmegaConf.to_container(cfg, resolve=False)
        fold_cfg = OmegaConf.create(fold_cfg)

        fold_cfg.data.loso_subject = test_subject
        print(test_subject)
        fold_cfg.experiment.name = f"{experiment_name}/loso_{test_subject}"

        train_main(fold_cfg)

        # Delete the checkpoint directory to save space, since we only care about the final results
        checkpoint_dir = f"{fold_cfg.experiment.name}/train/checkpoints"
        if Path(checkpoint_dir).exists():
            logging.info(f"Deleting checkpoint directory: {checkpoint_dir}")
            for file in Path(checkpoint_dir).glob("*"):
                file.unlink()
            Path(checkpoint_dir).rmdir()


if __name__ == '__main__':
    main()