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
import traceback

from omegaconf import OmegaConf

from src.utils.utils import load_config, get_experiment_directory, setup_logging

from scripts.train import main as train
from scripts.eval import main as evaluate
from scripts.predict import main as predict

import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    try:
        find_batch_size = cfg.experiment.find_batch_size
        min_batch_size = cfg.experiment.min_batch_size
        base_batch_size = cfg.experiment.base_batch_size
        base_learning_rate = cfg.experiment.base_learning_rate

        experiment_dir = get_experiment_directory(cfg)

        if find_batch_size:
            """
            Find the maximum batch size for memory, scale learning rate accordingly, and use gradient accumulation if necessary to simulate larger batch sizes.
            """
            batch_size = train(cfg, overwrite=True, find_batch_size=True)
            logging.info(f"Found maximum batch size: {batch_size}")

            if batch_size < min_batch_size:
                logging.warning(f"Found batch size {batch_size} is less than minimum {min_batch_size}. Gradient accumulation will be used.")
                num_accumulation_steps = (min_batch_size + batch_size - 1) // batch_size  # Equivalent to math.ceil(min_batch_size / batch_size)
                cfg.training.num_accumulation_steps = num_accumulation_steps
                logging.info(f"Set num_accumulation_steps to {num_accumulation_steps}")
                effective_batch_size = batch_size * num_accumulation_steps
            else:
                effective_batch_size = batch_size

            cfg.data.batch_size = batch_size

            # Adjust learning rate based on batch size (linear scaling rule)
            scaled_learning_rate = base_learning_rate * (effective_batch_size / base_batch_size)
            cfg.finetuning.learning_rate = scaled_learning_rate
            logging.info(f"Scaled learning rate to {scaled_learning_rate} based on batch size {effective_batch_size} and base batch size {base_batch_size}.")

        OmegaConf.save(cfg, experiment_dir / "config.yaml")

        train_results = train(cfg, overwrite=True, find_batch_size=False)
        logging.info(f"Training completed with results: {train_results}")

        if find_batch_size:
            # Optimization: Eval can usually handle 2x the training batch size
            cfg.data.batch_size = batch_size # * 2
            logging.info(f"Set evaluation batch size to {cfg.data.batch_size} based on discovered training batch size.")

        eval_results = evaluate(cfg, overwrite=True)
        logging.info(f"Evaluation completed with results: {eval_results}")

        # infer_results = predict(cfg, overwrite=overwrite)
        # logging.info(f"Inference completed with results: {infer_results}")

    except Exception as e:
        # Capture the full traceback
        error_msg = traceback.format_exc()
        logging.error(f"Job failed with error:\n{error_msg}")
        
        # Optional: Save error to a dedicated file in the exp folder
        # Note: only works if experiment_dir was successfully created
        try:
            with open(experiment_dir / "error.log", "w") as f:
                f.write(error_msg)
        except:
            pass
            
        # Re-raise if you want Hydra to acknowledge the failure
        raise e

    return {
        "train_results": train_results,
        "eval_results": eval_results,
        # "infer_results": infer_results
    }

if __name__ == "__main__":
    setup_logging()
    main()
