# EMG Foundation Model Benchmarking
Date created: 08/10/2025

## TL;DR

```bash
./setup.sh
python -m scripts.main +experiment=full_sweep
```

## Evironment setup

From virtual environment with python==3.12:

```bash
pip install -r requirements.txt
```

or using conda:

```bash
conda env create -f environment.yml
conda activate emg-fm-bench
```

To freeze:

```bash
pip list --format=freeze > requirements.txt
```


## Data Setup
All datasets are (optionally) downloaded and converted to a standardised HDF5 format. 
The `HDF5_ROOT` environment variable can optionally be used to define the location where datasets will be stored. 

```bash
# Set the dataset root
export HDF5_ROOT="./data"
# Download and convert datasets:

# NinaPro DB5
python -m scripts.datasets.ninapro_db5 --download_data --segment_gestures

# EPN-612
python -m scripts.datasets.epn612 --download_data --segment_gestures

# UCI-EMG
python -m scripts.datasets.uci_emg --download_data
```

Alternatively, the location of the original and converted datasets can be specified as command line arguments. 

### Command line arguments:

- `--dataset_dir`: The directory that the dataset is saved, or the directory to download to if `--download_data` is present
- `--download_data`: if present, original dataset is downloaded and saved to either dataset_dir, or ${HDF_ROOT} / <dataset_name> if --dataset_dir is not specified
- `--output_dir`: Directory to store the converted .hdf5 dataset. If absent, defaults to dataset_dir
- `--segment_gestures`: Flag to segment the original dataset by gesture label (NinaPro DB5 and EPN-612)
- `--include_rest`: Whether to segment and include the rest period between gestures as a "no_gesture" labelled signal (Ninapro DB5 Only)

## Download Pretrained Models

```bash
python -m scripts.download_checkpoints
```

## Run Experiments

### With Hyra
#### Single GPU
```bash
python -m scripts.main +experiment=full_sweep
```

#### Multi-GPU
```bash
python -m torch.distributed.run   --nproc_per_node=4 -m scripts.main +experiment=full_sweep
```



# References 
Soundstream implementation: https://github.com/haydenshively/SoundStream/tree/master
