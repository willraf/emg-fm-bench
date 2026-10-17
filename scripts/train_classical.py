""" Module train_classical.py

Training script for classical machine learning models for both
classification and regression tasks.

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
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr

import csv
import src.utils.visualisation as vis


# ------------------------------------------------------------------ #
# Model definitions
# ------------------------------------------------------------------ #

def get_classical_model(name: str, task: str = "classification"):
    if task == "classification":
        models = {
            "lda": LinearDiscriminantAnalysis(),
            "svm": SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr"),
            "rf":  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        }
    elif task == "regression":
        models = {
            # Ridge handles multi-output natively
            "ridge": Ridge(alpha=1.0),
            # SVR and RF need MultiOutputRegressor wrapper for multi-output
            "svr":   MultiOutputRegressor(SVR(kernel="rbf", C=10, gamma="scale"), n_jobs=-1),
            "rf":    RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        }
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'classification' or 'regression'.")

    assert name in models, f"Unknown model: '{name}' for task '{task}'. Choose from {list(models.keys())}"
    return models[name]


# ------------------------------------------------------------------ #
# Feature extraction
# ------------------------------------------------------------------ #

def hudgins_features(emg_window: torch.Tensor) -> np.ndarray:
    """
    Extract Hudgins time-domain features from a batch of EMG windows.

    Args:
        emg_window: (B, C, T) tensor

    Returns:
        features: (B, 4*C) numpy array
    """
    x = emg_window.cpu().numpy()   # (B, C, T)

    mav = np.mean(np.abs(x), axis=2)                                               # (B, C)
    wl  = np.sum(np.abs(np.diff(x, axis=2)), axis=2)                              # (B, C)

    threshold = 0.01
    sign = np.sign(x)
    sign[np.abs(x) < threshold] = 0
    zc  = np.sum(np.abs(np.diff(sign, axis=2)) > 0, axis=2).astype(float)        # (B, C)

    dx  = np.diff(x, axis=2)
    ssc = np.sum(np.abs(np.diff(np.sign(dx), axis=2)) > 0, axis=2).astype(float) # (B, C)

    return np.concatenate([mav, wl, zc, ssc], axis=1)                             # (B, 4*C)


def extract_features(
    dataloader: DataLoader,
    target_field: str,
    feature_fn=hudgins_features,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a full dataloader through a feature extraction function.

    Returns:
        X: (N, F) feature matrix
        y: (N,) for classification or (N, J) for regression
    """
    all_features = []
    all_labels   = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            emg     = batch["emg"]              # (B, C, T)
            targets = batch[target_field]       # (B,) or (B, J)

            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()

            features = feature_fn(emg)          # (B, F)
            all_features.append(features)
            all_labels.append(targets)

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels,   axis=0)

    if y.ndim > 2:
        y = y[:, -1, :]  # Take final frame of window for regression tasks

    return X, y


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def compute_classification_metrics(y_test, y_pred) -> dict:
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    return {
        "test_acc":       acc,
        "test_f1":        f1,
        "test_precision": precision,
        "test_recall":    recall,
    }


def compute_regression_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute per-joint then averaged regression metrics, consistent
    with RegressionMetrics in src/metrics.py.

    Args:
        y_test: (N, J)
        y_pred: (N, J)

    Returns:
        dict with mae, rmse, r2, pearson_r (all averaged across joints)
    """
    J = y_test.shape[1] if y_test.ndim > 1 else 1

    if y_test.ndim == 1:
        y_test = y_test[:, None]
        y_pred = y_pred[:, None]

    mae_per_joint      = np.mean(np.abs(y_test - y_pred), axis=0)             # (J,)
    rmse_per_joint     = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))     # (J,)

    # R² per joint — guards against zero-variance joints
    ss_res = ((y_test - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_test - y_test.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2_per_joint = np.where(ss_tot > 1e-8, 1.0 - ss_res / ss_tot, 0.0)       # (J,)

    # Pearson r per joint — guards against zero-variance
    pearson_per_joint = np.zeros(J)
    for j in range(J):
        if y_test[:, j].std() > 1e-8 and y_pred[:, j].std() > 1e-8:
            pearson_per_joint[j], _ = pearsonr(y_test[:, j], y_pred[:, j])

    return {
        "test_mae":       float(mae_per_joint.mean()),
        "test_rmse":      float(rmse_per_joint.mean()),
        "test_r2":        float(r2_per_joint.mean()),
        "test_pearson_r": float(pearson_per_joint.mean()),
    }


# ------------------------------------------------------------------ #
# Saving
# ------------------------------------------------------------------ #

def record_metrics(cfg, metrics: list[dict], save_dir, cm=None) -> None:
    fpath = save_dir / "results.csv"
    write_header = not fpath.exists()
    with open(fpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(metrics)

    if cm is not None and cfg.experiment.get("plot_examples", False):
        vis.plot_confusion_matrix(
            cm,
            class_names=cfg.data.get("class_names", None),
            fpath=save_dir / "figures" / "confusion_matrix.png"
        )


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def infer_task(cfg) -> str:
    """Infer task type from config, consistent with compile_results.py."""
    task_name = cfg.task.get("name", "")
    if "regression" in task_name:
        return "regression"
    return "classification"


def get_model_names(task: str) -> list[str]:
    if task == "classification":
        return ["lda", "svm", "rf"]
    return ["ridge", "svr", "rf"]


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg=None):
    seed = cfg.experiment.random_seed
    base_experiment_name = cfg.experiment.name
    save_experiment = cfg.experiment.save

    set_seed(seed)

    task = infer_task(cfg)
    logging.info(f"Task type: {task}")

    train_loader = setup_data(cfg.data, mode="train", distributed=False)
    test_loader  = setup_data(cfg.data, mode="test",  distributed=False)

    X_train, y_train = extract_features(train_loader, target_field=cfg.task.target_field)
    X_test,  y_test  = extract_features(test_loader,  target_field=cfg.task.target_field)

    logging.info(f"Train: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test:  X={X_test.shape},  y={y_test.shape}")

    for model_name in get_model_names(task):

        experiment_name = base_experiment_name.replace("model", model_name)
        cfg.experiment.name  = experiment_name
        cfg.model.name       = model_name
        cfg.model.fm.name    = model_name

        experiment_dir = get_experiment_directory(
            experiment_name=experiment_name, save_flag=save_experiment
        )
        setup_logging(experiment_dir=experiment_dir, is_main_process=True)
        eval_dir = setup_save_dir(
            cfg, experiment_dir=experiment_dir,
            overwrite=False, mode="eval", is_main_process=True
        )

        pipeline = Pipeline([
            ("scaler",     StandardScaler()),
            ("estimator",  get_classical_model(model_name, task=task)),
        ])

        logging.info(f"Fitting {model_name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Compute metrics
        if task == "classification":
            result = compute_classification_metrics(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
        else:
            result = compute_regression_metrics(y_test, y_pred)
            cm = None

        result["model"] = model_name
        logging.info(f"Results for '{model_name}': {result}")

        record_metrics(cfg, [result], eval_dir, cm=cm)


if __name__ == "__main__":
    main()