"""
Module visualisation

Contains methods for visualisation of EMG and related data

Usage:
    import src.utils.visualisation as vis
    vis.plot_emg(emg_data, fpath="path/to/save/plot.png")

Author:
    Will Raftery   

Date:
    31/07/2025

"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional

import torch 


def _convert_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch.Tensor to numpy array if needed."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def _save_or_show(fig: plt.Figure, fpath: Union[str, Path] = None) -> None:
    """Save figure to file if path provided, otherwise show it."""
    if fpath:
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath)
        logging.info(f"Saved plot to {fpath}")
    else:
        plt.show()
    plt.close(fig)


# EMG Signal Visualization
def plot_emg(emg: Union[np.ndarray, torch.Tensor], fpath: Optional[str] = None) -> None:
    """Plot EMG signals from multiple channels.
    
    Args:
        emg: EMG signals of shape (num_channels, time_steps)
        fpath: Optional path to save the plot
    """
    emg = _convert_to_numpy(emg)
    num_channels, time_steps = emg.shape
    
    fig, ax = plt.subplots(figsize=(50, 5))
    time = np.arange(time_steps)
    
    for ch in range(num_channels):
        ax.plot(time, emg[ch] + ch * 1, label=f'Channel {ch+1}')
    
    ax.set_title('EMG Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude + Offset')
    ax.grid(False)
    
    _save_or_show(fig, fpath)


def plot_reconstruction(emg: Union[np.ndarray, torch.Tensor], 
                        reconstruction: Union[np.ndarray, torch.Tensor], 
                        fpath: Optional[str] = None) -> None:
    """Plot EMG signals from multiple channels.
    
    Args:
        emg: EMG signals of shape (num_channels, time_steps)
        fpath: Optional path to save the plot
    """
    emg = _convert_to_numpy(emg)
    reconstruction = _convert_to_numpy(reconstruction)
    num_channels, time_steps = emg.shape
    
    fig, axes = plt.subplots(2, 1, figsize=(100, 5), sharey=True)
    time = np.arange(time_steps)
    
    # Plot original
    ax = axes[0]
    for ch in range(num_channels):
        ax.plot(time, emg[ch] + ch * 2, label=f'Channel {ch+1}')
    
    ax.set_title('EMG Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude + Offset')
    ax.grid(False)

    # Plot reconstruction
    ax = axes[1]
    for ch in range(num_channels):
        ax.plot(time, reconstruction[ch] + ch * 2, label=f'Channel {ch+1}')
    
    ax.set_title('Reconstruction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude + Offset')
    ax.grid(False)
    
    _save_or_show(fig, fpath)


# Training Metrics Visualization
def plot_loss_curves(train_loss: np.ndarray, val_loss: np.ndarray, 
                    epochs: np.ndarray, fpath: Optional[str] = None) -> None:
    """Plot training and validation loss curves.
    
    Args:
        train_loss: Array of training losses
        val_loss: Array of validation losses
        epochs: Array of epoch numbers
        fpath: Optional path to save the plot
    """
    train_loss = _convert_to_numpy(train_loss)
    val_loss = _convert_to_numpy(val_loss)
    epochs = _convert_to_numpy(epochs)

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line_train, = ax.plot([], [], 'r-', label='Train Loss')
    line_val, = ax.plot([], [], 'b-', label='Validation Loss')
    ax.legend()

    # Interim plot update
    line_train.set_xdata(epochs)
    line_train.set_ydata(train_loss)
    line_val.set_xdata(epochs)
    line_val.set_ydata(val_loss)
    ax.relim()
    ax.autoscale_view()


    
    _save_or_show(fig, fpath)


def plot_confusion_matrix(cm: Union[np.ndarray, torch.Tensor], 
                          class_names: Optional[list] = None, 
                          fpath: Optional[str] = None) -> None:
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix of shape (num_classes, num_classes)
        class_names: Optional list of class names for axes labels
        fpath: Optional path to save the plot
    """
    cm = _convert_to_numpy(cm)

    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig_size = (10, 8) if num_classes <= 10 else (num_classes, num_classes)
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is not None:
        ax.set(xticks=np.arange(num_classes), yticks=np.arange(num_classes),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix', ylabel='True label', xlabel='Predicted label')
    else:
        ax.set(title='Confusion Matrix', ylabel='True label', xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    if num_classes <= 10:    
        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        _save_or_show(fig, fpath)