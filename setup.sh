#!/bin/bash

# 1. Set HDF5_ROOT if not already set
if [ -z "$HDF5_ROOT" ]; then
    echo "HDF5_ROOT not set. Defaulting to ./data"
    export HDF5_ROOT="./data"
fi

# 2. Download and convert datasets
python -m scripts.datasets.ninapro_db5 --download_data --segment_gestures
python -m scripts.datasets.epn612 --download_data --segment_gestures
python -m scripts.datasets.uci_emg --download_data

# 3. Download pretrained model checkpoints
python -m scripts.download_checkpoints

echo "Setup complete. Run experiments with:"
echo "python -m scripts.main +experiment=full_sweep"
