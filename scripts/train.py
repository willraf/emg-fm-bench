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


# TODO remove generic utils import *
from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.models import setup_model, setup_discriminator

import torch

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


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

    optimizer_g = setup_optimiser(cfg, generator.parameters())
    optimizer_d = setup_optimiser(cfg, list(stft_disc.parameters()) + list(wave_disc.parameters()))

    loss_g, loss_d = setup_loss_function(cfg)

    results = {
        "epochs": [],   # Each epoch has different data (more like "meta-batches" after which validation is performed)
        "train_loss": [],
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
            loss_g,
            loss_d,
            epoch,
            log_interval=log_interval,
        )
        
        val_loss = validate(cfg,
            generator,
            stft_disc, 
            wave_disc, 
            val_loader, 
            loss_g,
            loss_d,  
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
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss) 
        results["best_val_loss"].append(best_val_loss)

        elapsed_time = time.time() - start_time 

        logging.info(f'====> Epoch: {epoch} Average train loss: {train_loss:.4f}, Average val loss: {val_loss:.4f}, Elapsed time: {elapsed_time:.2f} seconds')
        
        # Save results to file
        save_results(save_dir, epoch, train_loss, val_loss, best_val_loss)
        
        if plot_examples_flag:
            vis.plot_loss_curves(results["train_loss"], results["val_loss"], results["epochs"], fpath= save_dir / "figures" / "loss_curves.png")

    return results['best_val_loss'][-1]  # Return the best validation loss from the last epoch


def train(cfg, generator, stft_disc, wave_disc, train_loader, optimizer_g, optimizer_d, loss_g, loss_d, epoch, log_interval=None):
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
    generator.train()
    stft_disc.train()
    wave_disc.train()

    total_loss_g = 0.0
    total_loss_d = 0.0

    num_batches = len(train_loader)

    for batch_idx, data in enumerate(train_loader):

        inputs = data['emg'].to(device)
        inputs = inputs[:, :cfg.data.num_channels, :]  # Use only first channel
        inputs /= inputs.abs().max()

        # -----------------
        # Train Generator
        # -----------------
        vis.plot_emg(inputs[0], "figures/emg_input.png")

        optimizer_g.zero_grad()
        print(f'Inputs Shape: {inputs.shape}')
        reconstruction = generator(inputs)
        print(f'Reconstruction Shape: {reconstruction.shape}')
        wave_disc_outputs = wave_disc(reconstruction)
        print(f"Wave Disc Outputs shape: {[len(out) for out in wave_disc_outputs]}")
        print(f'Wave Disc Outputs[0][0] Shape: {wave_disc_outputs[0][0].shape}')
        print(f'Wave Disc Outputs[0][-1] Shape: {wave_disc_outputs[0][-1].shape}')
        stft_disc_inputs =  torch.stft(inputs.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 2, 1)
        print(f'STFT Inputs Shape: {stft_disc_inputs.shape}')
        stft_disc_outputs = stft_disc(stft_disc_inputs)
        print(f"STFT Disc Outputs shape: {[out.shape for out in stft_disc_outputs]}")
        exit()
        loss_generator = loss_g()
        loss_generator.backward()

        optimizer_g.step()

        total_loss_g += loss_generator.item()


        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_d.zero_grad()
        loss_discriminator = loss_d()

        loss_discriminator.backward()
        optimizer_d.step()
        total_loss_d += loss_discriminator.item()

        if log_interval is not None:
            # Prints the training loss on the current batch every `log_interval` batches.
            if batch_idx % log_interval == 0:
                logging.info(
                    "Train Epoch: {:02d} -- Batch: {:03d} -- Generator Loss: {:.4f} -- Discriminator Loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        loss_generator.item(),
                        loss_discriminator.item()
                    )
                )
        
        torch.cuda.empty_cache()

    # Note that the average loss is computed differently to the test funciton, 
    # but they both compute the same thing: the avereage loss over all datapoints in the dataset
    return{
        "loss_d": total_loss_d / num_batches,
        "loss_g": total_loss_g / num_batches
    } 

def validate(model, test_loader, loss_function, plot_examples=False, epoch=None, save_dir=None):
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

            inputs /= inputs.abs().max()


            outputs = model(inputs)

            loss = loss_function(outputs, inputs) #, reduction="mean")
            
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
