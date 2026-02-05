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
from src.models.losses import discriminator_loss, adversarial_loss, feature_loss, spectral_recon_loss

import torch

# Parallelisation
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os

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

    # Initialize distributed training
    init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    is_main = dist.get_rank() == 0

    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    logging.info("Starting training process...")

    # Set the seed for reproducibility
    set_seed(random_seed)
    logging.info(f"Random seed set to {random_seed}")

    save_dir = setup_save_dir(cfg, overwrite=overwrite, mode="train")
    logging.info(f'Experiment directory set up at: {save_dir}')

    if cfg.data.data_dir == "data":
        mode = "val"
    else:
        mode = "train"
    train_loader = setup_data(cfg, mode="val")
    val_loader = setup_data(cfg, mode="val")

    logging.info(f'Data loaders set up. Number of training batches: {len(train_loader)}, Number of validation batches: {len(val_loader)}')

    generator = setup_model(cfg, device=device)
    stft_disc, wave_disc = setup_discriminator(cfg, device=device)

    # Wrap models with DDP
    generator = DDP(generator, device_ids=[local_rank])
    stft_disc = DDP(stft_disc, device_ids=[local_rank])
    wave_disc = DDP(wave_disc, device_ids=[local_rank])

    # optimizer_g = setup_optimiser(cfg, generator.parameters())
    # optimizer_d = setup_optimiser(cfg, list(stft_disc.parameters()) + list(wave_disc.parameters()))

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(list(wave_disc.parameters()) + list(stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

    logging.info("Model and optimizers set up.")

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
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        start_time = time.time()

        train_loss = train(cfg, 
            generator,
            stft_disc, 
            wave_disc, 
            train_loader, 
            optimizer_g,
            optimizer_d,
            epoch,
            device=device,
            log_interval=log_interval,
        )
        
        val_loss = validate(cfg,
            generator,
            val_loader,
            device=device,
            plot_examples=plot_examples_flag,                
            epoch=epoch,
            save_dir=save_dir,
            is_main=is_main
        )

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / "checkpoints" / "best_model.pt"
            if is_main:
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
        if is_main:
            save_results(save_dir, epoch, train_loss, val_loss, best_val_loss)
        
            if plot_examples_flag:
                vis.plot_loss_curves(results["train_gen_loss"], results["val_loss"], results["epochs"], fpath= save_dir / "figures" / "loss_curves.png")

    return results['best_val_loss'][-1]  # Return the best validation loss from the last epoch


def train(cfg, generator, stft_disc, wave_disc, train_loader, optimizer_g, optimizer_d, epoch, device, log_interval=None):
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

        batch_start_time = time.time()
        logging.info(f'Starting batch {batch_idx} of {num_batches}...')

        inputs = data['emg'].to(device)
        inputs = inputs[:, :cfg.data.num_channels, :]  # Use only first channel
        # inputs /= inputs.abs().max()
        
        # -----------------
        # Train Generator
        # -----------------

        optimizer_g.zero_grad()

        reconstruction = generator(inputs)

        # generator_loss = torch.nn.MSELoss()(reconstruction, inputs) #, reduction="mean")
        # generator_loss.backward()
        # optimizer_g.step()
        # total_loss_g += generator_loss.item()
        # loss_discriminator = torch.tensor(0.0)  # Placeholder to avoid undefined variable error
        
        wave_disc_recon_outputs = wave_disc(reconstruction)
        wave_disc_real_outputs = wave_disc(inputs)

        # Reshape inputs to B*C for STFT function
        inputs_reshaped = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2))
        reconstruction_reshaped = reconstruction.view(reconstruction.size(0) * reconstruction.size(1), reconstruction.size(2))

        stft_disc_real_inputs = torch.stft(inputs_reshaped, n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 2, 1)
        stft_disc_real_outputs = stft_disc(stft_disc_real_inputs)

        stft_disc_recon_inputs =  torch.stft(reconstruction_reshaped, n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 2, 1)
        stft_disc_recon_outputs = stft_disc(stft_disc_recon_inputs)

        combined_real_fm = stft_disc_real_outputs + wave_disc_real_outputs
        combined_recon_fm = stft_disc_recon_outputs + wave_disc_recon_outputs

        disc_real_logits = [fm[-1] for fm in combined_real_fm]
        disc_recon_logits = [fm[-1] for fm in combined_recon_fm]

        reconstruction_loss = 10 * torch.nn.MSELoss()(reconstruction, inputs)
        a = adversarial_loss(disc_recon_logits)
        f = feature_loss(combined_real_fm, combined_recon_fm)
        s = spectral_recon_loss(inputs, reconstruction)
        
        generator_loss = reconstruction_loss + s + 100*a + 1000*f

        generator_loss.backward()

        optimizer_g.step()


        total_loss_g += generator_loss.item()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_d.zero_grad()

        wave_disc_real_outputs = wave_disc(inputs)
        stft_disc_real_outputs = stft_disc(stft_disc_real_inputs)

        wave_disc_recon_outputs = wave_disc(reconstruction.detach())
        stft_disc_recon_outputs = stft_disc(stft_disc_recon_inputs.detach())

        disc_real_logits_D = [stft_disc_real_outputs[-1]] + [w[-1] for w in wave_disc_real_outputs]
        disc_recon_logits_D = [stft_disc_recon_outputs[-1]] + [w[-1] for w in wave_disc_recon_outputs]

        # Calculate Discriminator Loss
        loss_discriminator = discriminator_loss(disc_real_logits_D, disc_recon_logits_D)
        
        # Backward pass only through the Discriminator's parameters
        torch.cuda.reset_peak_memory_stats()
        loss_discriminator.backward()
        optimizer_d.step()
        total_loss_d += loss_discriminator.item()
        max_mem = torch.cuda.max_memory_allocated() / 1e9
        logging.info(f'Maximum GPU memory allocated this batch (M_sample): {max_mem:.4f} GB')


        if log_interval is not None:
            # Prints the training loss on the current batch every `log_interval` batches.
            if batch_idx % log_interval == 0:
                logging.info(
                    "Train Epoch: {:02d} -- Batch: {:03d} -- Generator Loss: {:.4f} -- Discriminator Loss: {:.4f}".format(
                        epoch,
                        batch_idx,
                        generator_loss.item(),
                        loss_discriminator.item()
                    )
                )
                logging.info(
                    "Recon Loss: {:.4f} -- Adv Loss: {:.4f} -- Feature Loss: {:.4f} -- Spectral Loss: {:.4f}".format(
                        reconstruction_loss.item(),
                        a.item(),
                        f.item(),
                        s.item()
                    )
                )
        
        batch_time = time.time() - batch_start_time
        logging.info(f'----> Batch {batch_idx} processed in {batch_time:.2f} seconds')
        logging.info(f'Number of batches: {num_batches}')
        logging.info(f'Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.4f} GB')    
        exit()
        

    # Note that the average loss is computed differently to the test funciton, 
    # but they both compute the same thing: the avereage loss over all datapoints in the dataset
    return{
        "loss_d": total_loss_d / num_batches,
        "loss_g": total_loss_g / num_batches
    } 

def validate(cfg, model, test_loader, device, plot_examples=False, epoch=None, save_dir=None, is_main=False):
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
            inputs = inputs[:, :cfg.data.num_channels, :]  # Use only first channel

            # inputs /= inputs.abs().max()


            outputs = model(inputs)

            loss = torch.nn.MSELoss()(outputs, inputs) #, reduction="mean")
            
            val_loss += loss.item()

    if is_main:
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
