from src.tasks.base_task import BaseTask
from src.models import setup_model
import src.utils.visualisation as vis

from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import logging
from pathlib import Path



import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy import signal



# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def plot_reconstructions(originals_list, reconstructions_list, fs, bands,
                         labels=["NeuroRVQ"], save_dir="./figures"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    N, T = originals_list[0].shape
    time = np.linspace(0, T / fs, T)

    for i in tqdm(range(N), desc="Samples"):
        plt.figure(figsize=(10, 12))

        # Plot raw signals
        plt.subplot(6, 1, 1)
        orig = originals_list[0][i]
        recon = reconstructions_list[0][i]
        label = labels[0]

        plt.plot(time, orig, label=f"Original Signal", alpha=0.7)
        plt.plot(time, recon, linestyle='--', label=f"{label} Reconstruction", alpha=0.7)

        plt.title(f"Raw Signal")
        plt.legend()
        plt.ylabel("Amplitude")

        # Plot filtered bands
        for j, (band_name, (low, high)) in enumerate(bands.items()):
            plt.subplot(6, 1, j + 2)
            orig = originals_list[0][i]
            recon = reconstructions_list[0][i]
            label = labels[0]

            orig_band = bandpass_filter(orig, low, high, fs)
            recon_band = bandpass_filter(recon, low, high, fs)

            plt.plot(time, orig_band, label=f"{label} Original Signal", alpha=0.7)
            plt.plot(time, recon_band, linestyle='--', label=f"{label} Reconstruction", alpha=0.7)

            plt.title(f"{band_name} Band")
            plt.ylabel("Amplitude")

        plt.xlabel("Time (s)")
        plt.tight_layout()

        plt.savefig(f"{save_dir}/sample_{i}.png")
        plt.close()


def process_and_plot(originals, reconstructions, fs, mode):
    P, T = reconstructions[0].shape

    originals_np = [
        original.detach().cpu().numpy().reshape(P, T)
        for original in originals
    ]
    reconstructions_np = [
        reconstruction.detach().cpu().numpy().reshape(P, T)
        for reconstruction in reconstructions
    ]
    
    if mode=="EEG":
        # Define EEG bands
        bands = {
            "Delta (0.5–4 Hz)": (0.5, 4),
            "Theta (4–8 Hz)": (4, 8),
            "Alpha (8–13 Hz)": (8, 13),
            "Beta (13–30 Hz)": (13, 30),
            "Gamma (30–45 Hz)": (30, 45),
        }
    elif mode=="EMG":
        # Define EMG bands
        bands = {
            "Band 1 (20–60 Hz)": (20, 60),
            "Band 2 (60–125 Hz)": (60, 125),
            "Band 3 (125-200 Hz)": (125, 200),
            "Band 4 (200-250 Hz)": (200, 250),
            "Band 5 (250-400 Hz)": (250, 400),
        }

    # Plot
    plot_reconstructions(originals_np, reconstructions_np, fs, bands)
    

class NeuroRVQEmbeddingBuilder:

    """
    Helper class to build the temporal and spatial embedding for NeuroRVQ.
    """

    def __init__(self, max_patches):
        self.max_patches = max_patches
        self.global_channels = np.array([b'c1', b'c10', b'c11', b'c12', b'c13', b'c14', b'c15', b'c16',
                                        b'c2', b'c3', b'c4', b'c5', b'c6', b'c7', b'c8', b'c9'])

    def build(self, n_time, sample_channels):

        # temporal
        temp_ix = torch.arange(
            self.max_patches - n_time,
            self.max_patches
        ).repeat(len(sample_channels)).reshape(1, -1)

        # spatial
        spat_ix = torch.tensor([
            np.where(self.global_channels == c)[0][0]
            for c in sample_channels
        ])

        spat_ix = torch.repeat_interleave(spat_ix, n_time)
        spat_ix = spat_ix.reshape(1, -1)

        return temp_ix, spat_ix
    

class NeuroRVQEvalTask(BaseTask):
    def __init__(self, cfg, device, distributed=False, is_main=False, save_dir=None):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.distributed = distributed
        self.is_main = is_main
        self.save_dir = save_dir

        self.tokeniser = setup_model(cfg.model.tokeniser, device=device, model_name="neuro_rvq_tokenizer")
        self.tokeniser.to(device)
        tokeniser_path = cfg.model.tokeniser.checkpoint
        self.load_from_checkpoint(tokeniser_path, self.tokeniser)
        self.loss_fn = torch.nn.MSELoss()

        self.embed_builder = NeuroRVQEmbeddingBuilder(
            max_patches=cfg.model.fm.n_patches
        )

        ch_names = np.array([
            'c1','c2','c3','c4','c5','c6','c7','c8',
            'c9','c10','c11','c12','c13','c14','c15','c16'
        ])
        self.channel_names = np.array([c.lower().encode() for c in ch_names])

        # self.fm = setup_model(cfg.model.fm, model_name="neuro_rvq_fm")
        # self.fm.to(device)
        # fm_path = cfg.finetuning.fm_checkpoint
        # self.load_from_checkpoint(fm_path, self.fm)


    def train_step(self, model, batch_idx=None, epoch=None):
        return {"loss": 0.0}
        


    @torch.no_grad()
    def val_step(self, batch, batch_idx=None, epoch=None):

        batch = self.preprocess_batch(batch)

        emg = batch["emg"].to(self.device)  # (B, C, T)
        temporal_ix = batch["temporal_ix"].int().to(self.device)
        spatial_ix = batch["spatial_ix"].int().to(self.device)

        std_x, std_xrec = self.tokeniser(
            emg,
            temporal_embedding_ix=temporal_ix,
            spatial_embedding_ix=spatial_ix
        )

        if self.is_main and self.cfg.experiment.plot_examples and batch_idx == 0:
            plot_id = torch.randint(0, emg.shape[0], (1,)).item()
            process_and_plot(std_x, std_xrec, fs=1000, mode='EMG')

            for i in range(10):
                vis.plot_reconstruction(std_x[plot_id, i:i+1], std_xrec[plot_id, i:i+1], fpath=self.save_dir / "figures" / f'val_batch_{batch_idx}_epoch_{epoch}_{i}.png')
        
        loss = self.loss_fn(std_xrec, std_x).item()

        return {
            "loss": loss,
        }


    def eval_step(self, model, batch, device):
        return super().eval_step(model, batch, device)


    def load_from_checkpoint(self, checkpoint_path, model):
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        return checkpoint


    def load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
        
        checkpoint = torch.load(path)
        self.tokeniser.load_state_dict(checkpoint)
        return checkpoint


    def wrap_distributed(self, local_rank):
        self.tokeniser = DDP(self.tokeniser, device_ids=[local_rank], output_device=local_rank)
        self.fm = DDP(self.fm, device_ids=[local_rank], output_device=local_rank)



    def preprocess_batch(self, batch):
        emg = batch["emg"]  # (B, C, T)

        processed = []
        temporal_ix_list = []
        spatial_ix_list = []

        for x in emg:

            C, _ = x.shape
            n_time = self.cfg.model.fm.n_patches // C

            temp_ix, spat_ix = self.embed_builder.build(
                n_time,
                sample_channels=self.channel_names
            )

            # --- NeuroRVQ signal preprocessing ---
            target_fs = 1000
            highpass = 20
            lowpass = 400
            fs = 2000

            x_np = x.cpu().numpy()

            lowpass_applied = min(lowpass, fs/2) - 0.5
            b, a = signal.butter(
                N=3,
                Wn=[highpass, lowpass_applied],
                btype='bandpass',
                fs=fs
            )

            x_np = signal.filtfilt(b, a, x_np, axis=-1)

            if target_fs != fs:
                x_np = signal.resample(
                    x_np,
                    num=int(x_np.shape[-1] / fs * target_fs),
                    axis=-1
                )

            x_np = x_np.astype("float32")
            x_np = x_np[:, :3200]

            processed.append(torch.from_numpy(x_np))
            temporal_ix_list.append(temp_ix.squeeze(0))
            spatial_ix_list.append(spat_ix.squeeze(0))

        batch["emg"] = torch.stack(processed)
        batch["temporal_ix"] = torch.stack(temporal_ix_list)
        batch["spatial_ix"] = torch.stack(spatial_ix_list)

        return batch