""" Module train_classical.py

Training script for classical machine learning models (e.g., SVM, Random Forest, etc.)

Usage:


Authors:
    Will Raftery

Date:
    05/05/2026
"""

import torch
from torch.utils.data import DataLoader

import numpy as np
import argparse
import logging
from tqdm import tqdm
import hydra

from src.utils.utils import load_config, set_seed, get_experiment_directory, setup_logging, setup_save_dir
from src.data import setup_data

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import csv
import src.utils.visualisation as vis



def get_classical_model(name: str):
    models = {
        "lda": LinearDiscriminantAnalysis(),
        "svm": SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr"),
        "rf":  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }
    assert name in models, f"Unknown model: '{name}'. Choose from {list(models.keys())}"
    return models[name]


def hudgins_features(emg_window: torch.Tensor) -> np.ndarray:
    """
    Extract Hudgins time-domain features from a batch of EMG windows.

    Args:
        emg_window: (B, C, T) tensor — matches your BioFMWrapper input format

    Returns:
        features: (B, 4*C) numpy array
    """
    # Move to CPU numpy for feature computation
    x = emg_window.cpu().numpy()   # (B, C, T)

    # 1. Mean Absolute Value
    mav = np.mean(np.abs(x), axis=2)                                          # (B, C)

    # 2. Waveform Length
    wl = np.sum(np.abs(np.diff(x, axis=2)), axis=2)                           # (B, C)

    # 3. Zero Crossings (with dead zone to avoid noise crossings)
    threshold = 0.01
    sign = np.sign(x)
    sign[np.abs(x) < threshold] = 0
    zc = np.sum(np.abs(np.diff(sign, axis=2)) > 0, axis=2).astype(float)     # (B, C)

    # 4. Slope Sign Changes
    dx = np.diff(x, axis=2)                                                    # (B, C, T-1)
    ssc = np.sum(np.abs(np.diff(np.sign(dx), axis=2)) > 0, axis=2).astype(float)  # (B, C)

    return np.concatenate([mav, wl, zc, ssc], axis=1)                         # (B, 4*C)


def extract_features(dataloader: DataLoader, 
                     target_field: str,
                     feature_fn=hudgins_features,
                     device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """
    Run a full dataloader through a feature extraction function.

    Args:
        dataloader:     Your existing PyTorch DataLoader
        target_field:   Key for labels in batch dict e.g. "gesture_id"
        feature_fn:     Feature extraction function, default hudgins_features
        device:         Unused for classical features, kept for API consistency

    Returns:
        X: (N, F) feature matrix
        y: (N,)   label vector
    """
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            emg     = batch["emg"]                          # (B, C, T)
            targets = batch[target_field].numpy()           # (B,)

            features = feature_fn(emg)                     # (B, F)
            all_features.append(features)
            all_labels.append(targets)

    X = np.concatenate(all_features, axis=0)               # (N, F)
    y = np.concatenate(all_labels,   axis=0)               # (N,)

    return X, y


def record_metrics(cfg, metrics, cm, save_dir):
    fpath = save_dir / "results.csv"

    write_header = not fpath.exists()
    with open(fpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(metrics)

    if cfg.experiment.plot_examples:
        vis.plot_confusion_matrix(cm, class_names=cfg.data.get("class_names", None), fpath=save_dir / "figures" / "confusion_matrix.png")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    """
    Main provides an entry point for training a model based on a given configuration.
    It handles:
    - Set random seed
    - Logging
    - Setting up directory structures and saving
    - Training
    - Evaluation
    """
    # Experiment level setup
    seed = cfg.experiment.random_seed
    base_experiment_name = cfg.experiment.name
    save_experiment = cfg.experiment.save

    set_seed(seed)
    
    train_loader = setup_data(cfg.data, mode="train", distributed=False)
    test_loader = setup_data(cfg.data, mode="test", distributed=False)

    X_train, y_train = extract_features(train_loader, target_field=cfg.task.target_field)
    X_test, y_test = extract_features(test_loader, target_field=cfg.task.target_field)

    
    for model in ["lda", "svm", "rf"]:
        metrics = []
        experiment_name = base_experiment_name.replace("model", model)
        cfg.experiment.name = experiment_name
        cfg.model.name = model
        cfg.model.fm.name = model
        
        experiment_dir = get_experiment_directory(experiment_name=experiment_name, save_flag=save_experiment)
        setup_logging(experiment_dir=experiment_dir, is_main_process=True)
        eval_dir = setup_save_dir(cfg, experiment_dir=experiment_dir, overwrite=False, mode='eval', is_main_process=True)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", get_classical_model(model))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        result = {
            "model": model,
            "test_acc": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        }
        logging.info(result)
        metrics.append(result)

        record_metrics(cfg, metrics, cm, eval_dir)
        logging.info(f"Results for model '{model}'")
        logging.info(f"Test Accuracy: {acc:.4f}")
        logging.info(f"Test Precision: {precision:.4f}")
        logging.info(f"Test Recall: {recall:.4f}")
        logging.info(f"Test F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()

# if __name__ == "__main__":  
#     parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
#     parser.add_argument('--overwrite', action='store_true')
#     args = parser.parse_args()

#     cfg = load_config(args.config, default_path=args.default)
#     main(cfg, overwrite=args.overwrite)