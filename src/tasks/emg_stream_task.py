

import torch
import torch.optim as optim

from src.models import setup_model, setup_discriminator
from src.models.soundstream.losses import discriminator_loss, adversarial_loss, feature_loss, spectral_recon_loss
import src.utils.visualisation as vis

from torch.nn.parallel import DistributedDataParallel as DDP

import logging
from pathlib import Path

from src.tasks.base_task import BaseTask

class EMGStreamTask(BaseTask):
    def __init__(self, cfg, device, distributed=False, is_main=False, save_dir=None):
        
        self.cfg = cfg
        self.device = device
        self.distributed = distributed
        self.is_main = is_main
        self.save_dir = save_dir

        self.generator = setup_model(cfg, device=self.device)
        self.stft_disc, self.wave_disc = setup_discriminator(cfg, device=self.device)

        # Just incase
        self.generator.to(self.device)
        self.stft_disc.to(self.device)
        self.wave_disc.to(self.device)

        self.optimiser_g = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimiser_d = optim.Adam(list(self.wave_disc.parameters()) + list(self.stft_disc.parameters()), lr=1e-4, betas=(0.5, 0.9))

        self.stft_window = torch.hann_window(
            window_length=1024,
            device=self.device
        )

        self.recon_loss = torch.nn.MSELoss()

    def train_step(self, batch, batch_idx=None, epoch=None):
        """
        Must return at least a dictionary with 'loss' key for the main loss to be logged. Can include any other metrics as needed.
        """
        
        inputs = batch['emg'].to(self.device)

        # -----------------
        # Train Generator
        # -----------------
        # Freeze Discriminator parameters during Generator training
        for param in self.wave_disc.parameters():
            param.requires_grad = False
        for param in self.stft_disc.parameters():
            param.requires_grad = False

        reconstruction = self.generator(inputs)
        wave_disc_recon_out = self.wave_disc(reconstruction)
        wave_disc_real_out = self.wave_disc(inputs)

        # Reshape inputs to B*C for STFT function
        inputs_reshaped = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2))
        reconstruction_reshaped = reconstruction.view(reconstruction.size(0) * reconstruction.size(1), reconstruction.size(2))

        stft_disc_real_inputs = torch.stft(inputs_reshaped, n_fft=1024, hop_length=256, window=self.stft_window, return_complex=False).permute(0, 3, 2, 1)
        stft_disc_real_outputs = self.stft_disc(stft_disc_real_inputs)

        stft_disc_recon_inputs =  torch.stft(reconstruction_reshaped, n_fft=1024, hop_length=256, window=self.stft_window, return_complex=False).permute(0, 3, 2, 1)
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

        self.optimiser_g.zero_grad()
        generator_loss.backward()
        self.optimiser_g.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        # Unfreeze Discriminator parameters for Discriminator training
        for param in self.wave_disc.parameters():
            param.requires_grad = True
        for param in self.stft_disc.parameters():
            param.requires_grad = True


        wave_disc_real_outputs = self.wave_disc(inputs)
        stft_disc_real_outputs = self.stft_disc(stft_disc_real_inputs)

        wave_disc_recon_outputs = self.wave_disc(reconstruction.detach())
        stft_disc_recon_outputs = self.stft_disc(stft_disc_recon_inputs.detach())

        disc_real_logits_D = [stft_disc_real_outputs[-1]] + [w[-1] for w in wave_disc_real_outputs]
        disc_recon_logits_D = [stft_disc_recon_outputs[-1]] + [w[-1] for w in wave_disc_recon_outputs]

        # Calculate Discriminator Loss
        loss_discriminator = discriminator_loss(disc_real_logits_D, disc_recon_logits_D)
        
        # Backward pass only through the Discriminator's parameters
        self.optimiser_d.zero_grad()
        loss_discriminator.backward()
        self.optimiser_d.step()


        return {
            'loss': generator_loss.item(),
            'g_loss': generator_loss.item(),
            'd_loss': loss_discriminator.item(),
            'recon_loss': reconstruction_loss.item(),
            'adv_loss': a.item(),
            'feat_loss': f.item(),
            'spec_recon_loss': s.item(),
        }
    

    def val_step(self, batch, batch_idx=None, epoch=None):

        inputs = batch['emg'].to(self.device)
        outputs = self.generator(inputs)
        loss = self.recon_loss(outputs, inputs)
        
        if self.is_main and self.cfg.experiment.plot_examples and batch_idx == 0:
            plot_id = torch.randint(0, inputs.shape[0], (1,)).item()
            vis.plot_reconstruction(inputs[plot_id], outputs[plot_id], fpath=self.save_dir / "figures" / f'val_batch_{batch_idx}_epoch_{epoch}.png')
        
        return {
            'loss': loss.item(),
        }


    def wrap_distributed(self, local_rank):
        self.generator = DDP(self.generator, device_ids=[local_rank])
        self.stft_disc = DDP(self.stft_disc, device_ids=[local_rank])
        self.wave_disc = DDP(self.wave_disc, device_ids=[local_rank])    


    def load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
        
        checkpoint = torch.load(path)

        # Load model states
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.stft_disc.load_state_dict(checkpoint['discriminator_stft_state_dict'])
        self.wave_disc.load_state_dict(checkpoint['discriminator_wave_state_dict'])

        # Load optimizer states (optional, needed only to resume training)
        if 'optimizer_G_state_dict' in checkpoint:
            self.optimiser_g.load_state_dict(checkpoint['optimizer_G_state_dict'])
        if 'optimizer_D_state_dict' in checkpoint:
            self.optimiser_d.load_state_dict(checkpoint['optimizer_D_state_dict'])

        logging.info(f"Checkpoint loaded successfully from {path} (Epoch {checkpoint['epoch']})")
        return checkpoint


    def save_checkpoint(self, checkpoint_path, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            
            # Generator components
            'generator_state_dict': self.unwrap_model(self.generator).state_dict(),
            'optimizer_G_state_dict': self.optimiser_g.state_dict(),
            
            # Discriminator components (saved separately for clarity)
            'discriminator_stft_state_dict': self.unwrap_model(self.stft_disc).state_dict(),
            'discriminator_wave_state_dict': self.unwrap_model(self.wave_disc).state_dict(),
            'optimizer_D_state_dict': self.optimiser_d.state_dict(),
        }

        path = Path(checkpoint_path)
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved: {path}")

    
    def modules(self):
        return [self.generator, self.stft_disc, self.wave_disc]