""" Module tsne.py

Visualise the t-SNE embedding of the training data.

Usage:


Authors:
    Will Raftery

Date:
    26/03/2026
"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.utils import *
from src.data import setup_data
from src.models import setup_model
import torch
import numpy as np
import argparse

def main(cfg):
    distributed, device, rank, local_rank = setup_distributed()
    val_loader = setup_data(cfg.data, mode="val", distributed=distributed)

    fm = setup_model(cfg.model.fm, device=device)
    fm.to(device)
    fm.eval()

    all_embeddings = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            emg = batch['emg'].to(device)
            targets = batch['gesture_id'].long().to(device)  # (B,)

            embeddings = fm(emg) # (B, num_temporal_patches, Channels, Embed_Dim)
            B, C, N, D = embeddings.shape

            # Pool embeddings
            embeddings = embeddings.permute(0, 2, 1, 3).reshape(B, N, -1).mean(dim=1) # (B, Chan*Embed)

            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches into one large array
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 1. Reduce dimensions to 50 using PCA (Standard practice)
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(all_embeddings)

    # 2. Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    vis_dims = tsne.fit_transform(embeddings_pca)

    # Assuming you collected 'all_labels' in your loop alongside 'all_embeddings'
    plot_tsne(vis_dims, all_targets)


def plot_tsne(vis_dims, labels, title="t-SNE of Biosignal Embeddings"):
    plt.figure(figsize=(10, 7))
    
    # Using Seaborn for cleaner aesthetics and automatic legends
    sns.scatterplot(
        x=vis_dims[:, 0], 
        y=vis_dims[:, 1], 
        hue=labels, 
        palette='viridis',
        legend='full',
        alpha=0.6
    )
    
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    setup_logging()
    
    main(cfg)