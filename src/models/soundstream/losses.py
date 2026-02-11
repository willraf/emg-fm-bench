import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

import matplotlib.pyplot as plt

def discriminator_loss(disc_real_logits, disc_recon_logits):
    """
    Args:
        disc_real_logits: List[Tensor]
            - Outer list over discriminators
            - Each Tensor shape: (B, 1, T) - discriminator output for real data
        disc_recon_logits: List[Tensor]
            - Outer list over discriminators
            - Each Tensor shape: (B, 1, T) - discriminator output for reconstructed data
    """
    loss_real = 0
    loss_recon = 0
    for real_logits, recon_logits in zip(disc_real_logits, disc_recon_logits):
        loss_real += torch.mean(torch.relu(1.0 - real_logits))
        loss_recon += torch.mean(torch.relu(1.0 + recon_logits))
    loss_real /= len(disc_real_logits)
    loss_recon /= len(disc_recon_logits)
    return loss_real + loss_recon

def adversarial_loss(disc_recon_logits):
    loss = 0
    for recon_logits in disc_recon_logits:
        loss += torch.mean(torch.relu(1.0 - recon_logits))
    loss /= len(disc_recon_logits)
    return loss


def feature_loss(real_fm, recon_fm):
    """
    Args:
        real_fm: List[List[Tensor]]
            - Outer list over discriminators
            - Inner list over layers
            - Each Tensor shape: (B, C, T) - feature map from a layer for real data
        recon_fm: List[List[Tensor]]
            - Outer list over discriminators
            - Inner list over layers
            - Each Tensor shape: (B, C, T) - feature map from a layer for reconstructed data
    """
    loss = 0
    num_discriminators = len(real_fm)
    for disc_real_fm, disc_recon_fm in zip(real_fm, recon_fm):
        num_layers = len(disc_real_fm)
        for real_feat, recon_feat in zip(disc_real_fm, disc_recon_fm):
            loss += torch.mean(torch.abs(real_feat - recon_feat))
        loss /= num_layers
    loss /= num_discriminators
    return loss

# def spectral_recon_loss(x, G_x, eps=1e-5):
#     device = x.device
#     L = 0
#     for i in range(6,12):
#         s = 2**i
#         alpha_s = (s/2)**0.5
#         melspec = MelSpectrogram(sample_rate=2000, n_fft=s, hop_length=s//4, n_mels=8, wkwargs={"device": device}).to(device)
#         S_x = melspec(x)
#         S_G_x = melspec(G_x)
        
#         loss = (S_x-S_G_x).abs().sum() + alpha_s*(((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).sum(dim=-2)**0.5).sum()
#         L += loss
    
#     return L


def spectral_recon_loss(x, x_hat):
    """
    Calculates the Multi-Scale Spectral Reconstruction Loss for the generator.

    Args:
        x (Tensor): The original waveform (B, in_channels, T) or (B, T)
        x_hat (Tensor): The reconstructed waveform (B, in_channels, T) or (B, T)

    Returns:
        Tensor: The scalar loss value.
    """
    scales = [64, 128, 256, 512, 1024] #, 2048]  # 2^6 to 2^11
    total_loss = 0.0

    # Ensure waveforms are 2D (B, T)
    if x.dim() == 3:
        # Flatten (B, C, T) to (B*C, T) to process each channel as an independent sample
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        x_hat = x_hat.reshape(B * C, T)

    for s in scales:
        hop_length = s // 4
        alpha_s = torch.sqrt(torch.tensor(s / 2.0)).to(x.device)

        S_real = torch.stft(x, n_fft=s, hop_length=hop_length, window=torch.hann_window(window_length=s, device=x.device), return_complex=True)
        S_hat = torch.stft(x_hat, n_fft=s, hop_length=hop_length, window=torch.hann_window(window_length=s, device=x.device), return_complex=True)

        S_real = S_real.abs()
        S_hat = S_hat.abs()

        # 2. L1 Loss on Magnitude (Linear Scale)
        # ||S_real - S_hat||_1 averaged over all elements
        loss_l1 = F.l1_loss(S_hat, S_real) 

        # 3. L2 Loss on Log Magnitude (Logarithmic/Perceptual Scale)
        # Use a small epsilon (1e-5) for numerical stability before taking log
        log_S_real = torch.log(S_real + 1e-5)
        log_S_hat = torch.log(S_hat + 1e-5)
        
        # ||log(S_real) - log(S_hat)||_2^2 averaged over all elements
        loss_l2 = F.mse_loss(log_S_hat, log_S_real) 

        # 4. Total loss for this scale (s)
        total_loss += loss_l1 + alpha_s * loss_l2

    # Return the sum over all scales
    return total_loss