# EMG Foundation Model Benchmarking
Date created: 08/10/2025

# Evironment setup

From virtual environment:

`$ pip install -r requirements.txt`

To freeze:

`$ pip list --format=freeze > requirements.txt`


# Data Setup

To download an process datasets:

## EPN612

`python scripts/datasets/epn.py --dataset_dir data/epn612 --download_data`

## NinaPro DB5

`python scripts/datasets/ninapro_db5.py --dataset_dir data/ninapro_db5 --download_data`


# References 
Soundstream implementation: https://github.com/haydenshively/SoundStream/tree/master