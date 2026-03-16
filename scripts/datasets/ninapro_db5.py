""" Script ninapro_db5.py

Convert Ninapro DB5 from its original .mat format into a structured HDF5 format for efficient loading in our EMG processing pipeline.

NOTE:
    this script segments signals into individual gesture trials based on the 'restimulus' and 'rerepetition' labels.
    Rest portions (stimulus=0) are ignored in the final dataset, so final gesture IDs range from 1-52

Usage:


Authors:
    Will Raftery

Date:
    19/02/2026
"""



def main():
    pass

if __name__ == '__main__':
    main()


import h5py
import scipy.io
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import urllib.request
import zipfile
import io

TRAIN_SUBJS = [1, 3, 4, 6, 8, 9, 10]
VAL_SUBJS = [11]
TEST_SUBJS = [5, 7]

TRAIN_REPS = [1, 3, 4, 6]
VAL_REPS = [2]
TEST_REPS = [5]

INCLUDE_REST = False

def get_gesture_id(exercise_id, stimulus_id):
    """NinaPro DB5 has a unique gesture distribution across exercises:
    
    Each exercise has a different set of gestures:

    Exercise 1 (2 in fname, B in docs): 17 gestures (1-17)
    Exercise 2 (3 in fname, C in docs): 23 gestures (18-40)
    Exercise 3 (1 in fname, A in docs): 12 gestures (41-52)
    """

    if stimulus_id == 0: return 0 # Rest is always 0
    
    # Offsets based on the 17, 23, 12 distribution
    if exercise_id == 1:
        return stimulus_id
    elif exercise_id == 2:
        return stimulus_id + 17
    elif exercise_id == 3:
        return stimulus_id + 17 + 23
    return stimulus_id

def get_global_gesture_id(gesture_id):
    """Convert dataset-specific gesture IDs to a global standard for multi-dataset training
    currently left as placeholder"""
    return gesture_id

def process_metadata(mat):
    """Standardize NinaPro metadata to match DATA_SPEC.md"""
    # Standardize gender
    gender = str(mat['gender'][0]).lower()
    if 'm' in gender: gender = 'm'
    elif 'f' in gender: gender = 'f'
    else: gender = 'u'

    # Standardize laterality -> handedness
    lat = str(mat['laterality'][0]).lower()
    handedness = 'r' if 'r' in lat else ('l' if 'l' in lat else 'u')

    standard_user = {
        'subject_id': f"S{int(mat['subject'][0][0])}",
        'age': int(mat['age'][0][0]),
        'gender': gender,
        'handedness': handedness,
        'weight_kg': float(mat['weight'][0][0]),
        'height_cm': float(mat['height'][0][0]),
        'forearm_circumference_cm': float(mat['circumference'][0][0]),
    }
    
    standard_session = {
        'sample_rate': int(mat['frequency'][0][0]),
        'exercise_id': int(mat['exercise'][0][0])
    }
    
    return standard_user, standard_session

def download_ninapro_db5(data_dir):
    """
    Downloads and flattens the NinaPro DB5 structure.
    Official URL: https://ninapro.hevs.ch/files/DB5_Preproc/
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://ninapro.hevs.ch/files/DB5_Preproc/"
    
    for i in range(1, 11):  # Subjects 1-10
        url = f"{base_url}s{i}.zip"
        print(f"Downloading subject {i} from {url}...")
        
        try:
            # Download into memory to avoid leaving zip artifacts
            with urllib.request.urlopen(url) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as zip_ref:
                    # Flattening logic:
                    for member in zip_ref.namelist():
                        filename = os.path.basename(member)
                        # Skip directories and non-mat files
                        if not filename or not filename.endswith('.mat'):
                            continue
                        
                        # Extract the file directly to data_dir
                        source = zip_ref.open(member)
                        target = open(data_dir / filename, "wb")
                        with source, target:
                            target.write(source.read())
            print(f"Successfully extracted subject {i} .mat files to {data_dir}")
            
        except Exception as e:
            print(f"Error downloading subject {i}: {e}")


def convert_ninapro_db5(dataset_dir, output_dir=None, download_data=False):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        output_fpath = dataset_dir / "ninapro_db5.hdf5"
    else:
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_fpath = output_dir / "ninapro_db5.hdf5"

    # Should work with both nested and slat dir structures
    mat_files = sorted(dataset_dir.rglob("*.mat"))

    if download_data and len(mat_files) == 0:
        print("No .mat files found. Attempting to download data...")
        download_ninapro_db5(dataset_dir)
        mat_files = sorted(dataset_dir.glob("*.mat"))
        if len(mat_files) == 0:
            print("Download failed or no .mat files found after download. Please check the dataset directory.")
            return
    elif len(mat_files) == 0:
        print("No .mat files found in the dataset directory. Please provide the raw data or enable download with --download_data.")
        return

    with h5py.File(output_fpath, 'w') as f:
        root = f.create_group("ninapro_db5")
        
        for fpath in tqdm(mat_files, desc="Converting NinaPro DB5"):
            mat = scipy.io.loadmat(str(fpath))

            user_std, sess_std = process_metadata(mat)
            
            subj_key = user_std['subject_id']
            if subj_key not in root:
                subj_group = root.create_group(subj_key)
                for k, v in user_std.items():
                    subj_group.attrs[k] = v

                subject_id_int = int(mat['subject'][0][0])
                if subject_id_int in TRAIN_SUBJS:
                    subj_split = 'train'
                elif subject_id_int in VAL_SUBJS:
                    subj_split = 'val'
                elif subject_id_int in TEST_SUBJS:
                    subj_split = 'test'
                else:
                    print(f"Warning: Subject {subj_key} not found in predefined splits. Assigning 'unknown' split.")
                    subj_split = 'unknown'

                subj_group.attrs['inter_subject_split'] = subj_split
                    
            else:
                subj_group = root[subj_key]

            # Continuous data arrays
            emg = mat['emg'].astype(np.float32) / 128.0 # Standardize to [-1, 1]
            acc = mat['acc'].astype(np.float32)
            glove = mat['glove'].astype(np.float32)
            
            # Use 'restimulus' and 'rerepetition' for refined labels
            stimulus = mat['restimulus'].flatten()
            repetition = mat['rerepetition'].flatten()

            # 2. Identify transitions to catch "Rest" periods
            # A "trial" is now defined as any continuous block with the same stimulus ID
            diff = np.diff(stimulus, prepend=-1)
            trial_starts = np.where(diff != 0)[0]
            trial_ends = np.append(trial_starts[1:], len(stimulus))

            for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
                stim_idx = stimulus[start]
                rep_idx = repetition[start]

                if not INCLUDE_REST and stim_idx == 0:
                    continue
                
                # We still ignore very short windows
                # NOTE: includnig rest is likely to cause label leakage, as distribution of signal lengths is very different for rest vs active periods.
                if (end - start) < 200: 
                    continue
                
                exercise_id = sess_std['exercise_id']
                gesture_id = get_gesture_id(exercise_id, stim_idx)

                # 3. Handle naming for Rest (Label 0)
                # We add a unique counter 'T' for Trial because Repetition 0 isn't unique
                sample_name = f"E{exercise_id}_T{i}_G{gesture_id}"
                sample_group = subj_group.create_group(sample_name)
                
                # Metadata
                sample_group.attrs['gesture_id'] = gesture_id
                sample_group.attrs['repetition_id'] = rep_idx


                if rep_idx == 2:  # Validation repetition
                    sample_group.attrs['intra_session_split'] = 'val'
                elif rep_idx == 5:  # Test repetition
                    sample_group.attrs['intra_session_split'] = 'test'
                else:
                    sample_group.attrs['intra_session_split'] = 'train'

                sample_group.attrs['inter_session_split'] = 'n/a'  # NinaPro DB5 doesn't have multiple sessions per subject
                sample_group.attrs['inter_subject_split'] = subj_group.attrs.get('inter_subject_split', 'unknown')
                
                for k, v in sess_std.items():
                    sample_group.attrs[k] = v

                # All modalities are oversampled to 200Hz, so they should be aligned by index
                
                # 1. EMG (T, 16) 
                sample_group.create_dataset("emg", data=emg[start:end], compression="lzf")
                sample_group["emg"].attrs['sample_rate'] = sess_std.get('sample_rate', 200)
                sample_group["emg"].attrs['units'] = 'normalized'
                sensor_val = mat.get('sensor', ['unknown'])[0]
                # Use np.string_ to ensure it's HDF5-compatible (fixed-width)
                sample_group["emg"].attrs['sensor_type'] = str(sensor_val)
                
                # 2. Accel (T, 3)
                sample_group.create_dataset("accel", data=acc[start:end], compression="lzf")
                sample_group["accel"].attrs['sample_rate'] = sess_std.get('sample_rate', 200)
                sample_group["accel"].attrs['sensor_type'] = str(sensor_val)
                
                # 3. Glove (T, 22) - Kinematic ground truth
                sample_group.create_dataset("glove", data=glove[start:end], compression="lzf")
                sample_group["glove"].attrs['sample_rate'] = sess_std.get('sample_rate', 200)
                sample_group["glove"].attrs['sensor_type'] = 'cyber_glove'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/ninapro_db5")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--download_data", action='store_true', help="If set, will attempt to download the raw .mat files from the official Ninapro DB5 source")
    args = parser.parse_args()
    
    convert_ninapro_db5(args.dataset_dir, args.output_dir, args.download_data)