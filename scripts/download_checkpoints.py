""" Script download_checkpoints.py

Downloads pretrained model checkpoints for EMG-FM-Bench.

Usage:
    python -m scripts.download_checkpoints                    # download all models
    python -m scripts.download_checkpoints --models tinymyo neurorvq  # download specific models

Date:
    30/06/2026
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PRETRAINED_DIR = Path("pretrained_models")

# ------------------------------------------------------------------ #
# TinyMyo
# ------------------------------------------------------------------ #

def download_tinymyo():
    """
    Downloads TinyMyo from HuggingFace Hub.
    Repo: https://huggingface.co/MatteoFasulo/TinyMyo
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    output_dir = PRETRAINED_DIR / "TinyMyo"
    if output_dir.exists() and any(output_dir.iterdir()):
        logging.info(f"TinyMyo already exists at {output_dir}, skipping.")
        return

    logging.info("Downloading TinyMyo from HuggingFace...")
    snapshot_download(
        repo_id="MatteoFasulo/TinyMyo",
        repo_type="model",
        local_dir=str(output_dir),
    )
    logging.info(f"TinyMyo downloaded to {output_dir}")


# ------------------------------------------------------------------ #
# NeuroRVQ
# ------------------------------------------------------------------ #

# NEURORVQ_FILES = [
#     "pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt",
#     "pretrained_models/foundation_models/NeuroRVQ_EEG_foundation_model_v1.pt",
#     "pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt",
#     "pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt",
#     "pretrained_models/tokenizers/NeuroRVQ_ECG_tokenizer_v1.pt",
#     "pretrained_models/foundation_models/NeuroRVQ_ECG_foundation_model_v1.pt",
#     "pretrained_models/tokenizers/NeuroRVQ_PPG_tokenizer_v1.pt",
# ]

NEURORVQ_FILES = [
    "pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt",
    "pretrained_models/tokenizers/NeuroRVQ_EMG_tokenizer_v1.pt",
]

def download_neurorvq():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    output_dir = PRETRAINED_DIR / "NeuroRVQ"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading NeuroRVQ from HuggingFace...")

    for filename in NEURORVQ_FILES:
        # filename in repo: "pretrained_models/foundation_models/NeuroRVQ_EMG_..."
        # we want it saved to: pretrained_models/NeuroRVQ/foundation_models/...
        # so strip the "pretrained_models/" prefix from the repo path
        relative_path = Path(filename).relative_to("pretrained_models")
        local_path = output_dir / relative_path

        if local_path.exists():
            logging.info(f"Already exists: {local_path}, skipping.")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Downloading {filename} → {local_path}")
        downloaded = hf_hub_download(
            repo_id="ntinosbarmpas/NeuroRVQ",
            filename=filename,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,  # force actual copy, not symlink
        )

        # hf_hub_download may save with the full repo path structure
        # move to our desired location if needed
        downloaded_path = Path(downloaded)
        if downloaded_path != local_path and downloaded_path.exists():
            downloaded_path.rename(local_path)
            logging.info(f"Moved to {local_path}")

    logging.info(f"NeuroRVQ downloaded to {output_dir}")

# NEURORVQ_FILES = [
#     "pretrained_models/foundation_models/NeuroRVQ_EMG_foundation_model_v1.pt"
# ]

# def download_neurorvq():
#     """
#     Downloads NeuroRVQ checkpoints from HuggingFace Hub.
#     Repo: https://huggingface.co/ntinosbarmpas/NeuroRVQ
#     """
#     try:
#         from huggingface_hub import hf_hub_download
#     except ImportError:
#         raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

#     logging.info("Downloading NeuroRVQ from HuggingFace...")
#     for filename in NEURORVQ_FILES:
#         # Check if already downloaded — hf_hub_download saves relative to local_dir
#         local_path = PRETRAINED_DIR / "NeuroRVQ" / filename
#         local_path = PRETRAINED_DIR.parent / filename
#         if local_path.exists():
#             logging.info(f"Already exists: {local_path}, skipping.")
#             continue

#         logging.info(f"Downloading {filename}...")
#         hf_hub_download(
#             repo_id="ntinosbarmpas/NeuroRVQ",
#             filename=filename,
#             local_dir=str(PRETRAINED_DIR.parent),  # saves to ./pretrained_models/NeuroRVQ/...
#         )

#     logging.info(f"NeuroRVQ downloaded to {PRETRAINED_DIR / 'NeuroRVQ'}")


# ------------------------------------------------------------------ #
# PhysioWave
# ------------------------------------------------------------------ #

PHYSIOWAVE_GDRIVE_ID = "1pioRsvzrJDTELyRF51iIpYcvKP0WIyzP"
PHYSIOWAVE_FILENAME  = "emg.pth"

def download_physiowave():
    """
    Downloads PhysioWave checkpoint from Google Drive.
    
    NOTE: Google Drive does not support direct programmatic download of large files
    without authentication. If gdown fails, download manually from:
    https://drive.google.com/file/d/1pioRsvzrJDTELyRF51iIpYcvKP0WIyzP/view?usp=drive_link
    and place in pretrained_models/PhysioWave/physiowave_emg.pt
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for PhysioWave download. "
            "Install with: pip install gdown\n"
            "Alternatively, download manually from: "
            "https://drive.google.com/file/d/1pioRsvzrJDTELyRF51iIpYcvKP0WIyzP/view?usp=drive_link"
        )

    output_dir = PRETRAINED_DIR / "PhysioWave"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / PHYSIOWAVE_FILENAME

    if output_path.exists():
        logging.info(f"PhysioWave already exists at {output_path}, skipping.")
        return

    logging.info("Downloading PhysioWave from Google Drive...")
    url = f"https://drive.google.com/uc?id={PHYSIOWAVE_GDRIVE_ID}"
    gdown.download(url, str(output_path), quiet=False)
    logging.info(f"PhysioWave downloaded to {output_path}")


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #

MODELS = {
    "tinymyo":    download_tinymyo,
    "neurorvq":   download_neurorvq,
    "physiowave": download_physiowave,
}


def download_all():
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for name, fn in MODELS.items():
        try:
            fn()
        except Exception as e:
            logging.error(f"Failed to download {name}: {e}")
            errors.append(name)

    if errors:
        logging.warning(
            f"The following models failed to download: {errors}\n"
            f"Please download them manually and place in {PRETRAINED_DIR}/<ModelName>/"
        )
    else:
        logging.info("All checkpoints downloaded successfully.")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained model checkpoints.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to download. Defaults to all.",
    )
    args = parser.parse_args()

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if "all" in args.models:
        download_all()
    else:
        for model_name in args.models:
            try:
                MODELS[model_name]()
            except Exception as e:
                logging.error(f"Failed to download {model_name}: {e}")