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


STAGE_TO_IDX = {
    'ThumbsUpDownThumbRotationsCWCCWP': 0,
    'HandClawGraspFlicks': 1,
    'ShakaVulcanPeace': 2,
    'HookEmHornsOKScissors': 3,
    'TwoHandedFreeStyle': 4,
    'unconstrained': 5,
    'OneHandedFreeStyle': 6,
    'CountingUpDownFaceSideAway': 7,
    'ThumbsSwipesWholeHand': 8,
    'FingerWigglingSpreading': 9,
    'FingerAbductionSeries': 10,
    'FingerPinchesSingleFingerPinchesMultiple': 11,
    'FingerFreeform': 12,
    'IndividualFingerPointingSnap': 13,
    'GraspPunchCloseFar': 14,
    'PokeDrawPinchRotateclosefar': 15,
    'AllFingerPinchesThumbSwipeThumbRotate': 16,
    'CountingUpDownFingerWigglingSpreading': 17,
    'IndexPinchesMiddlePinchesThumbswipes': 18,
    'DoorknobFingerGraspFistGrab': 19,
    'FastPongFronthandBackhandThrowing': 20,
    'HandDeskSeparateClaspedChest': 21,
    'PokePinchCloseFar': 22,
    'FingerTouchPalmClapmrburns': 23,
    'WristFlexionAbduction': 24,
    'PlayBlocksChess': 25,
    'CoffeePanicPete': 26,
    'HandOverHandAllFingerPinchesThumbSwipeThumbRotate': 27,
    'HandOverHandCountingUpDownFingerWigglingSpreading': 28

}



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

        self.stage_to_idx = STAGE_TO_IDX

        self.embed_builder = NeuroRVQEmbeddingBuilder(
            max_patches=cfg.model.fm.n_patches
        )

        ch_names = np.array([
            'c1','c2','c3','c4','c5','c6','c7','c8',
            'c9','c10','c11','c12','c13','c14','c15','c16'
        ]) 
        self.channel_names = np.array([c.lower().encode() for c in ch_names])

        self.fm = setup_model(cfg.model.fm, model_name="neuro_rvq_fm", device=device)

        fm_path = cfg.model.fm.checkpoint
        # self.load_from_checkpoint(fm_path, self.fm)

        n_outputs = 29 # Number of stages
        self.fm.reset_classifier(n_outputs)

        self.fm.to(device)

        if cfg.training.freeze_backbone:
            # Freeze everything
            for p in self.fm.parameters():
                p.requires_grad = False

            # Unfreeze classifier head
            for p in self.fm.head.parameters():
                p.requires_grad = True

            for p in self.fm.fc_norm.parameters():
                p.requires_grad = True


            head_params = list(self.fm.head.parameters()) + \
                list(self.fm.fc_norm.parameters())

            self.optimiser = torch.optim.AdamW(
                head_params,
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay
            )
        else:
            self.optimiser = torch.optim.AdamW(
                self.fm.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay
            )

        self.loss_fn = torch.nn.CrossEntropyLoss()



    def train_step(self, batch, batch_idx=None, epoch=None):
        temporal_ix = batch["temporal_ix"].int().to(self.device)
        spatial_ix = batch["spatial_ix"].int().to(self.device)
        emg = batch["emg"].to(self.device)  # (B, C, T)
        stage = batch["stage"].to(self.device)  # (B,)

        B, C, T = emg.shape
        emg = emg.reshape(B, C, 16, 200)

        logits, _ = self.fm(
            emg,
            temporal_embedding_ix=temporal_ix,
            spatial_embedding_ix=spatial_ix
        )

        loss = self.loss_fn(logits, stage)
        
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        self.optimiser.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == stage).float().mean()

        return {
            "loss": loss.item(),
            "acc": acc.item()
        }        


    @torch.no_grad()
    def val_step(self, batch, batch_idx=None, epoch=None):

        emg = batch["emg"].to(self.device)  # (B, C, T)
        temporal_ix = batch["temporal_ix"].int().to(self.device)
        spatial_ix = batch["spatial_ix"].int().to(self.device)
        stage = batch["stage"].to(self.device)  # (B,)
        B, C, T = emg.shape
        emg = emg.reshape(B, C, 16, 200)
        logits, _ = self.fm(
            emg,
            temporal_embedding_ix=temporal_ix,
            spatial_embedding_ix=spatial_ix
        )
        loss = self.loss_fn(logits, stage)
        preds = logits.argmax(dim=1)
        acc = (preds == stage).float().mean()

        return {
            "loss": loss.item(),
            "acc": acc.item()
        }


    def eval_step(self, model, batch, device):
        return 


    def load_from_checkpoint(self, checkpoint_path, model):
        path = Path(checkpoint_path)
        if not path.exists():
            logging.warning(f"Checkpoint not found: {path}")
            return {}
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        return checkpoint


    def save_checkpoint(self, checkpoint_path, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.fm.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")


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
