import argparse
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import time
from pathlib import Path
import csv
import os
from collections import defaultdict


# TODO remove generic utils import *
from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.tasks import setup_task
from src.models import setup_model

import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import wandb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

TARGET_FIELD = "participant_id"  # Change this to the appropriate field in your dataset


def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings from a model given a dataloader.
    """
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            inputs = batch["emg"].to(device) 

            # Forward pass through the model to get embeddings
            embedding = model(inputs)
            embeddings.append(embedding.cpu())
            labels.append(batch[TARGET_FIELD].cpu()) 

    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(labels, axis=0)
    )



def knn_probe(model, train_loader, test_loader, device, k=5, normalise_embeddings=True):
    # Extract embeddings
    train_embeddings, train_labels = extract_embeddings(
        model, train_loader, device
    )
    test_embeddings, test_labels = extract_embeddings(
        model, test_loader, device
    )  

    print(train_embeddings.shape, train_labels.shape)  

    # Mean pooling
    train_embeddings = np.mean(train_embeddings, axis=(1, 2))       
    test_embeddings = np.mean(test_embeddings, axis=(1, 2))

    print(train_embeddings.shape, train_labels.shape)

    if normalise_embeddings:
        train_embeddings = normalize(train_embeddings, norm="l2")
        test_embeddings = normalize(test_embeddings, norm="l2")

    # Fit KNN and evaluate
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine" if normalise_embeddings else "euclidean",
        n_jobs=-1
    )
    knn.fit(train_embeddings, train_labels)
    accuracy = knn.score(test_embeddings, test_labels)
    
    return accuracy

def main(cfg, overwrite=False, find_batch_size=False):
    """
    Main provides an entry point for training a model based on a given configuration.
    It handles:
    - Set random seed
    - Distributed
    - Logging & wandb
    - Setting up directory structures and saving
    - Trainer init and running
    - Tear down
    """

    # Experiment level setup
    seed = cfg.experiment.random_seed
    experiment_name = cfg.experiment.name
    save_experiment = cfg.experiment.save
    ddp_debug_mode = cfg.experiment.get("ddp_debug_mode", False)
    use_wandb = cfg.experiment.get("use_wandb", False)
    wandb_project = cfg.experiment.get("wandb_project", "emg-foundation-model")

    # 1. Set the random seed
    set_seed(seed)

    # # 2. Setup distributed training
    # distributed, device, rank, local_rank = setup_distributed(ddp_debug_mode=ddp_debug_mode)
    # is_main = (rank == 0)
    device = "cpu"
    distributed = False
    is_main = True

    # 3. setup logging and results, config, figures etc. saving directory
    experiment_dir = get_experiment_directory(experiment_name=experiment_name, save_flag=save_experiment)
    setup_logging(experiment_dir=experiment_dir, is_main_process=is_main)
    
    train_dir = setup_save_dir(cfg, experiment_dir=experiment_dir, overwrite=overwrite, mode='train', is_main_process=is_main)

    model = setup_model(cfg.model.fm, device=device)
    fm_path = cfg.model.fm.checkpoint
    fm_path = Path(fm_path)

    if fm_path.exists():
        logging.info(f"Loading foundation model from checkpoint: {fm_path}")
        model.load_checkpoint(fm_path)
    else:
        # logging.warning(f"Foundation model checkpoint not found: {fm_path}. Training from scratch instead.")
        raise FileNotFoundError(f"Foundation model checkpoint not found: {fm_path}. Please provide a valid checkpoint or set 'from_scratch' to True.")

    train_loader = setup_data(cfg.data, mode="train", distributed=distributed)
    val_loader = setup_data(cfg.data, mode="val", distributed=distributed)
    test_loader = setup_data(cfg.data, mode="test", distributed=distributed)

    accuracy = knn_probe(model, train_loader, test_loader, device)

    print(accuracy)



if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--find_batch_size', action='store_true', required=False, default=False, help="Run batch size finder instead of training")
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    main(cfg, overwrite=args.overwrite, find_batch_size=args.find_batch_size)