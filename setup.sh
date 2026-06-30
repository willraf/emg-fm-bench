#!/bin/bash
# setup.sh — one-time environment setup for EMG-FM-Bench

# 1. Create conda environment
# conda env create -f environment.yml
# conda activate emg-fm-bench

conda env create -n emg-fm-bench python=3.12
conda activate emg-fm-bench
pip install -r requirements.txt

# 2. Set HDF5_ROOT if not already set
if [ -z "$HDF5_ROOT" ]; then
    echo "HDF5_ROOT not set. Defaulting to ./data"
    export HDF5_ROOT="./data"
fi

# 3. Download and convert datasets
python -m scripts.datasets.ninapro_db5 --download_data --segment_gestures
python -m scripts.datasets.epn612 --download_data --segment_gestures
python -m scripts.datasets.uci_emg --download_data

# 4. Download pretrained model checkpoints
python -m scripts.download_checkpoints

echo "Setup complete. Run experiments with:"
echo "python -m scripts.main +experiment=full_sweep"