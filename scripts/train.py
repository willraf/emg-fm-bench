"""
Script train

Handles model training and validation.

Usage:
    $ python -m scripts.train --config configs.train.yaml
    $ python -m scripts.train --config configs.train.yaml --default configs/default.yaml
    For distributed training:
    $ python -m torch.distributed.run   --nproc_per_node=1 -m scripts.train --config configs/main.yaml

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
import os


# TODO remove generic utils import *
from src.utils.utils import *
import src.utils.visualisation as vis
from src.data import setup_data
from src.models import setup_model, setup_discriminator
from src.models.losses import discriminator_loss, adversarial_loss, feature_loss, spectral_recon_loss

import torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class Trainer:
    """
    Trainer class for (potentially distributed) training.
    Responsible for training, validation, checkpointing, and metric logging.

    Attributes:
        cfg (OmegaConf): Configuration object
        overwrite (bool): Whether to overwrite existing experiment directory
        distributed (bool): Whether distributed training is enabled
        device (torch.device): Device to run training on ("cpu" , "cuda", "cuda:<local_rank>")
        rank (int): Global rank of the current process
        local_rank (int): Local rank of the current process
        is_main (bool): Whether this is the main process (rank 0)
        save_dir (Path): Directory to save checkpoints and results
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        model (torch.nn.Module): Model to be trained
        optimiser (torch.optim.Optimizer): Optimiser for training
        loss_fn (callable): Loss function
        scheduler (optional): Curriculum scheduler
             For changing data difficulty over epochs, usually for synthetic data.
        epochs_run (int): Number of epochs already run (for resuming training)
        metrics (list): List of dicts recording training metrics per epoch
    """

    def __init__(self, cfg, overwrite=False):

        set_seed(cfg.experiment.random_seed)

        self.cfg = cfg
        self.overwrite = overwrite        
        # Setup GPU device and distributed training
        self.distributed, self.device, self.rank, self.local_rank = setup_distributed()
        print(f"Process {self.rank} using device {self.device}, distributed={self.distributed}, local_rank={self.local_rank}")
        world_size = dist.get_world_size() if self.distributed else 1
        print(f"World size: {world_size}")
        self.is_main = (self.rank == 0)

        self.save_dir = setup_save_dir(cfg, overwrite=overwrite, mode='train', is_main_process=self.is_main)

        self.train_loader = setup_data(cfg, mode="train", distributed=self.distributed)
        self.val_loader = setup_data(cfg, mode="val", distributed=self.distributed)
        
        self.generator = setup_model(cfg, device=self.device)
        self.stft_disc, self.wave_disc = setup_discriminator(cfg, device=self.device)

        self.optimiser_g = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimiser_d = optim.Adam(list(self.wave_disc.parameters()) + list(self.stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

        self.epochs_run = 0
        
        if cfg.model.checkpoint is not None:
            if os.path.isfile(cfg.model.checkpoint):
                checkpoint = load_checkpoint(cfg.model.checkpoint, self.model, self.optimiser)
                self.epochs_run = checkpoint['epoch']

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.metrics = []

    def train(self):
        """
        Main training loop
        """
        num_epochs = self.cfg.training.epochs
        patience = self.cfg.training.patience
        log_interval = self.cfg.logging.log_interval
        plot_examples_flag = self.cfg.experiment.plot_examples

        best_val_loss = float('inf')
        epochs_no_improve = 0
        stop_training = False

        for epoch in range(self.epochs_run, num_epochs):

            train_loss = self._run_epoch(epoch)
            val_loss = self._val(epoch)

            if self.distributed:
                val_loss = self._reduce_mean(val_loss)

            logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best Val Loss: {best_val_loss:.6f}")
            self._record_metrics(epoch, train_loss, val_loss, best_val_loss)

            # Early stopping and best model saving
            if self.rank == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                
                    checkpoint_path = self.save_dir / "checkpoints" / "best_model.pt"
                    save_checkpoint(self.model, self.optimiser, epoch, val_loss, checkpoint_path)
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    stop_training = True

            if self.distributed:
                stop_tensor = torch.tensor(int(stop_training), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break


        logging.info("Training complete.")


    def _run_epoch(self, epoch):

        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        self.generator.train()
        self.stft_disc.train()
        self.wave_disc.train()

        total_loss_g = 0.0
        total_loss_d = 0.0

        num_batches = len(self.train_loader)

        for batch_idx, data in enumerate(self.train_loader):
            inputs = data['emg'].to(self.device)
            inputs = inputs[:, :cfg.data.num_channels, :]
            
            loss_g, loss_d = self._run_batch(inputs, batch_idx=batch_idx, epoch=epoch)
            total_loss_g += loss_g
            total_loss_d += loss_d
        
        return total_loss_g / num_batches
    

    def _run_batch(self, inputs, batch_idx=None, epoch=None):
        log_interval = self.cfg.logging.log_interval

        # -----------------
        # Train Generator
        # -----------------
        self.optimiser_g.zero_grad()
        reconstruction = self.generator(inputs)
        wave_disc_recon_out = self.wave_disc(reconstruction)
        wave_disc_real_out = self.wave_disc(inputs)

        # Reshape inputs to B*C for STFT function
        inputs_reshaped = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2))
        reconstruction_reshaped = reconstruction.view(reconstruction.size(0) * reconstruction.size(1), reconstruction.size(2))

        stft_disc_real_inputs = torch.stft(inputs_reshaped, n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=self.device), return_complex=False).permute(0, 3, 2, 1)
        stft_disc_real_outputs = self.stft_disc(stft_disc_real_inputs)

        stft_disc_recon_inputs =  torch.stft(reconstruction_reshaped, n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=self.device), return_complex=False).permute(0, 3, 2, 1)
        stft_disc_recon_outputs = self.stft_disc(stft_disc_recon_inputs)

        combined_real_fm = stft_disc_real_outputs + wave_disc_real_out
        combined_recon_fm = stft_disc_recon_outputs + wave_disc_recon_out

        disc_real_logits = [fm[-1] for fm in combined_real_fm]
        disc_recon_logits = [fm[-1] for fm in combined_recon_fm]

        reconstruction_loss = 10 * torch.nn.MSELoss()(reconstruction, inputs)
        a = adversarial_loss(disc_recon_logits)
        f = feature_loss(combined_real_fm, combined_recon_fm)
        s = spectral_recon_loss(inputs, reconstruction)
        
        generator_loss = reconstruction_loss + s + 100*a + 1000*f

        generator_loss.backward()
        self.optimiser_g.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimiser_d.zero_grad()

        wave_disc_real_outputs = self.wave_disc(inputs)
        stft_disc_real_outputs = self.stft_disc(stft_disc_real_inputs)

        wave_disc_recon_outputs = self.wave_disc(reconstruction.detach())
        stft_disc_recon_outputs = self.stft_disc(stft_disc_recon_inputs.detach())

        disc_real_logits_D = [stft_disc_real_outputs[-1]] + [w[-1] for w in wave_disc_real_outputs]
        disc_recon_logits_D = [stft_disc_recon_outputs[-1]] + [w[-1] for w in wave_disc_recon_outputs]

        # Calculate Discriminator Loss
        loss_discriminator = discriminator_loss(disc_real_logits_D, disc_recon_logits_D)
        
        # Backward pass only through the Discriminator's parameters
        loss_discriminator.backward()
        self.optimiser_d.step()

        if self.is_main and log_interval is not None:
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

        return generator_loss.item(), loss_discriminator.item()
    
    def _val(self, epoch):
        self.generator.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                inputs = data['emg'].to(self.device)
                inputs = inputs[:, :cfg.data.num_channels, :]

                outputs = self.generator(inputs)
                loss = torch.nn.MSELoss()(outputs, inputs)

                total_loss += loss.item()
        
        if self.is_main and self.cfg.experiment.plot_examples:
            plot_id = torch.randint(0, inputs.shape[0], (1,)).item()
            vis.plot_reconstruction(inputs[plot_id], outputs[plot_id], fpath=self.save_dir / "figures" / f'val_epoch_{epoch}.png')

        return total_loss / num_batches
    

    def _record_metrics(self, epoch, train_loss, val_loss, best_val_loss):
        """
        Record training metrics from a single epoch. Save to file if this is the main process.
        """
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss
        }
        self.metrics.append(record)

        if self.rank == 0:
            fpath = self.save_dir / "results.csv"

            write_header = not fpath.exists()
            with open(fpath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

            if self.cfg.experiment.plot_examples:
                train_losses = [m['train_loss'] for m in self.metrics]
                val_losses = [m['val_loss'] for m in self.metrics]
                epochs = [m['epoch'] for m in self.metrics]
                vis.plot_loss_curves(train_losses, val_losses, epochs, fpath=self.save_dir / "figures" / "loss_curves.png")


    def _reduce_mean(self, value):
        """Get the average of a tensor over all processes."""
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        return tensor.item()
    

    def find_max_batch_size(self, start_bs=32, max_trials=10, safety_factor=0.8) -> int:
        """
        Utility function to find the maximum batch size that fits in GPU memory.
        Uses a binary search approach, starting from `start_bs` and doubling until OOM, then refining.

        Args:
            start_bs (int): Initial batch size to try
            max_trials (int): Maximum number of trials to find the optimal batch size
            safety_factor (float): Factor to reduce the found batch size by for safety margin
        Returns:
            int: Maximum batch size that fits in GPU memory with safety margin
        """

        if not self.is_main:
            return None

        logging.info("\n🔍 Starting batch size probe...\n")

        batch_size = start_bs
        last_good = start_bs

        for trial in range(max_trials):
            logging.info(f"Trial {trial+1}/{max_trials}: Testing batch size {batch_size}...")
            cfg.data.batch_size = batch_size
            loader = setup_data(cfg, mode="val")
            
            data = next(iter(loader))

            inputs = data['emg'].to(self.device)

            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                self._run_batch(inputs, 1, 1)

                peak = torch.cuda.max_memory_allocated() / 1e9
                logging.info(f"Success | Peak memory {peak:.2f} GB")

                last_good = batch_size
                batch_size *= 2

            except RuntimeError as e:

                if "out of memory" in str(e).lower():
                    logging.info(f"OOM at batch size {batch_size}")
                    break
                else:
                    raise

        recommended = max(1, int(last_good * safety_factor))
        logging.info(f"\nRecommended per-GPU batch size: {recommended}")

        return recommended
    

def main(cfg, overwrite=False):
    
    trainer = Trainer(cfg, overwrite=overwrite)
    # trainer.find_max_batch_size()
    trainer.train()
    if trainer.distributed:
        destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a YAML config file")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--default', type=str, required=False, default="configs/default.yaml")
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config, default_path=args.default)
    setup_logging()
    
    main(cfg, overwrite=args.overwrite)
