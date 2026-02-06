"""
Script train

Handles model training and validation.

Usage:
    $ python -m scripts.train --config configs.train.yaml
    $ python -m scripts.train --config configs.train.yaml --default configs/default.yaml

Author: 
    Will Raftery

Date:
    30/07/2025
"""

import argparse
import logging
from omegaconf import OmegaConf
from tqdm import tqdm 
import time
from pathlib import Path
import csv


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm


# TODO remove generic utils import *
from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.models import setup_model, setup_discriminator
from src.models.losses import discriminator_loss, adversarial_loss, feature_loss, spectral_recon_loss

import torch

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')



def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    
    stft_loss = F.relu(1-features_stft_disc_G_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    wave_loss = torch.cat([F.relu(1-features_wave_disc_G_x[key][-1]).sum(dim=2).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names])
    loss = torch.cat([stft_loss, wave_loss]).mean()
    
    return loss

def feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft):
    wave_disc_names = lengths_wave.keys()
    
    stft_loss = torch.stack([((feat_x-feat_G_x).abs().sum(dim=-1)/lengths_stft[i].view(-1,1,1)).sum(dim=-1).sum(dim=-1) for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1).mean(dim=1, keepdim=True)
    wave_loss = torch.stack([torch.stack([(feat_x-feat_G_x).abs().sum(dim=-1).sum(dim=-1)/lengths_wave[key][i] for i, (feat_x, feat_G_x) in enumerate(zip(features_wave_disc_x[key], features_wave_disc_G_x[key]))], dim=1) for key in wave_disc_names], dim=2).mean(dim=1)
    loss = torch.cat([stft_loss, wave_loss], dim=1).mean()
    
    return loss

def spectral_reconstruction_loss(x, G_x, eps=1e-4):
    L = 0
    for i in range(6,12):
        s = 2**i
        alpha_s = (s/2)**0.5
        melspec = MelSpectrogram(sample_rate=2000, n_fft=s, hop_length=s//4, n_mels=8, wkwargs={"device": device}).to(device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        
        loss = (S_x-S_G_x).abs().sum() + alpha_s*(((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).sum(dim=-2)**0.5).sum()
        L += loss
    
    return L

def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    
    real_stft_loss = F.relu(1-features_stft_disc_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    real_wave_loss = torch.stack([F.relu(1-features_wave_disc_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1,1), real_wave_loss], dim=1).mean()
    
    generated_stft_loss = F.relu(1+features_stft_disc_G_x[-1]).sum(dim=-1).squeeze()/lengths_stft[-1].squeeze()
    generated_wave_loss = torch.stack([F.relu(1+features_wave_disc_G_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    generated_loss = torch.cat([generated_stft_loss.view(-1,1), generated_wave_loss], dim=1).mean()
    
    return real_loss + generated_loss

def save_results(save_dir, epoch, train_loss, val_loss, best_val_loss):
    """
    Append training results to file
    """
    train_results_fpath = Path(save_dir) / "results.csv"
    
    with open(train_results_fpath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, best_val_loss])


def main(cfg, overwrite=False):

    random_seed = cfg.experiment.random_seed
    patience = cfg.training.patience
    log_interval = cfg.logging.log_interval
    num_epochs = cfg.training.epochs
    plot_examples_flag = cfg.experiment.plot_examples

    # setup basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set the seed for reproducibility
    set_seed(random_seed)

    save_dir = setup_save_dir(cfg, overwrite=overwrite, mode="train")

    train_loader = setup_data(cfg, mode="val")
    val_loader = setup_data(cfg, mode="val")

    generator = setup_model(cfg, device=device)
    stft_disc, wave_disc = setup_discriminator(cfg, device=device)

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

    results = {
        "epochs": [],   # Each epoch has different data (more like "meta-batches" after which validation is performed)
        "train_disc_loss": [],
        "train_gen_loss": [],
        "val_loss": [],
        "best_val_loss": []
    }

    # For early stopping and saving best model
    best_val_loss = float('inf')

    counter = 0
    
    # Train-test loop
    for epoch in tqdm(range(num_epochs)):
        
        start_time = time.time()

        train_loss = train(cfg, 
            generator,
            stft_disc, 
            wave_disc, 
            train_loader, 
            optimizer_g,
            optimizer_d,
            epoch,
            log_interval=log_interval,
        )
        
        val_loss = validate(cfg,
            generator,
            val_loader,  
            plot_examples=plot_examples_flag,                
            epoch=epoch,
            save_dir=save_dir,
        )

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / "checkpoints" / "best_model.pt"
            save_checkpoint(generator, stft_disc, wave_disc, optimizer_g, optimizer_d, epoch, val_loss, checkpoint_path)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break


        # For plotting
        results["epochs"].append(epoch)
        results["train_disc_loss"].append(train_loss["loss_d"])
        results["train_gen_loss"].append(train_loss["loss_g"])
        results["val_loss"].append(val_loss) 
        results["best_val_loss"].append(best_val_loss)

        elapsed_time = time.time() - start_time 

        logging.info(f'====> Epoch: {epoch} Average train generator loss: {train_loss['loss_g']:.4f}, Average train discriminator loss: {train_loss['loss_d']:.4f}, Average val loss: {val_loss:.4f}, Elapsed time: {elapsed_time:.2f} seconds')
        
        # Save results to file
        save_results(save_dir, epoch, train_loss, val_loss, best_val_loss)
        
        if plot_examples_flag:
            vis.plot_loss_curves(results["train_gen_loss"], results["val_loss"], results["epochs"], fpath= save_dir / "figures" / "loss_curves.png")

    return results['best_val_loss'][-1]  # Return the best validation loss from the last epoch


def train(cfg, generator, stft_disc, wave_disc, train_loader, optimizer_g, optimizer_d, epoch, log_interval=None):
    """
    A utility function that performs a basic training loop.

    For each batch in the training set, fetched using `train_loader`:
        - Zeroes the gradient used by `optimizer`
        - Performs forward pass through `model` on the given batch
        - Computes loss on batch
        - Performs backward pass
        - `optimizer` updates model parameters using computed gradient

    Prints the training loss on the current batch every `log_interval` batches.
    """
    criterion_g = lambda x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft: LAMBDA_ADV*adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave) + LAMBDA_FEAT*feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft) + LAMBDA_REC*spectral_reconstruction_loss(x, G_x)
    criterion_d = adversarial_d_loss

    generator.train()
    stft_disc.train()
    wave_disc.train()

    total_loss_g = 0.0
    total_loss_d = 0.0

    num_batches = len(train_loader)

    for batch_idx, data in enumerate(train_loader):

        inputs = data['emg'].to(device)
        x = inputs[:, :cfg.data.num_channels, :]  # Use only first channel

        x = x.to(device)
        lengths_x = torch.tensor([x.shape[2]]*x.shape[0], device=device)
        
        G_x = generator(x)
        
        s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
        s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
        
        lengths_stft = stft_disc.features_lengths(lengths_s_x)
        lengths_wave = wave_disc.features_lengths(lengths_x)
        
        features_stft_disc_x = stft_disc(s_x)
        features_wave_disc_x = wave_disc(x)
        
        features_stft_disc_G_x = stft_disc(s_G_x)
        features_wave_disc_G_x = wave_disc(G_x)
        
        loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
        total_loss_g += loss_g.item()
        
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        
        features_stft_disc_x = stft_disc(s_x)
        features_wave_disc_x = wave_disc(x)
        
        features_stft_disc_G_x_det = stft_disc(s_G_x.detach())
        features_wave_disc_G_x_det = wave_disc(G_x.detach())
        
        loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
        
        total_loss_d += loss_d.item()
        
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        if log_interval is not None:
            # Prints the training loss on the current batch every `log_interval` batches.
            if batch_idx % log_interval == 0:
                logging.info(
                    "Train Epoch: {:02d} -- Batch: {:03d} -- Generator Loss: {:.4f} -- Discriminator Loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        loss_g.item(),
                        loss_d.item()
                    )
                )
        
    # Note that the average loss is computed differently to the test funciton, 
    # but they both compute the same thing: the avereage loss over all datapoints in the dataset
    return{
        "loss_d": total_loss_d / num_batches,
        "loss_g": total_loss_g / num_batches
    } 

def validate(cfg, model, test_loader, plot_examples=False, epoch=None, save_dir=None):
    """
    A utility function to compute the loss and accuracy on a test set by
    iterating through the test set using the provided `test_loader` and
    accumulating the loss and accuracy on each batch.       
    """
    val_loss = 0.0

    num_batches = len(test_loader)

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data['emg'].to(device)
            inputs = inputs[:, :1, :]  # Use only first channel

            # inputs /= inputs.abs().max()


            outputs = model(inputs)

            loss = torch.nn.MSELoss()(outputs, inputs) #, reduction="mean")
            
            val_loss += loss.item()

    if plot_examples:
        plot_id = torch.randint(0, inputs.shape[0], (1,)).item()
        vis.plot_reconstruction(inputs[plot_id], outputs[plot_id], fpath=save_dir / "figures" / f'val_epoch_{epoch}.png')

    # Divide by total number of samples to get average loss
    val_loss /= num_batches

    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
        
    main(cfg, overwrite=args.overwrite)
