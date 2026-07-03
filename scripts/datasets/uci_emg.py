""" Module uci_emg.py

Script to (optionally download and) process the UCI EMG dataset to HDF5 for use in biosignal analysis and machine learning tasks.

Usage:
    $ python -m scripts.datasets.uci_emg --dataset_dir data/uci_emg/ --output_dir data/uci_emg/ --download_data


Authors:
    Will Raftery

Date:
    25/03/2026

"""

import h5py
import json
import numpy as np
import argparse
import urllib.request
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
import os


TRAIN_USERS = [1, 3, 5, 6, 7, 8, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22, 23, 26, 27, 28, 30, 31, 32, 34, 35, 36]
VAL_USERS = [4, 13, 25, 33]
TEST_USERS =[2, 9, 14, 17, 24, 29]

assert len(set(TRAIN_USERS) & set(VAL_USERS)) == 0, "Overlap between train and val users"
assert len(set(TRAIN_USERS) & set(TEST_USERS)) == 0, "Overlap between train and test users"
assert len(set(VAL_USERS) & set(TEST_USERS)) == 0, "Overlap between val and test users"
assert len(set(TRAIN_USERS) | set(VAL_USERS) | set(TEST_USERS)) == 36, "Total unique users should be 36"


TRAIN_REPS = [2, 4]
# Not enough reps to do a full 3-way split
VAL_REPS = [1]
TEST_REPS = [3]

# assert len(set(TRAIN_REPS) & set(VAL_REPS)) == 0, "Overlap between train and val reps"
# assert len(set(TRAIN_REPS) & set(TEST_REPS)) == 0, "Overlap between train and test reps"
# assert len(set(VAL_REPS) & set(TEST_REPS)) == 0, "Overlap between val and test reps"
# assert len(set(TRAIN_REPS) | set(VAL_REPS) | set(TEST_REPS)) == 25, "Total unique reps should be 25"


SEGMENT_GESTURES = True
INCLUDE_UNKNOWN_LABELS = False
# 7 - extended palm (the gesture was not performed by all subjects).
INCLUDE_EXTENDED_PALM = False

ORIGINAL_GESTURE_MAPPING = {
    0: "unmarked data",
    1: "hand at rest", 
    2: "hand clenched in a fist",
    3: "wrist flexion",
    4: "wrist extension",
    5: "radial deviations",
    6: "ulnar deviations",
    7: "extended palm (the gesture was not performed by all subjects)"
}

NEW_GESTURE_MAPPING = {k-1: v for k, v in ORIGINAL_GESTURE_MAPPING.items()}

def download_dataset(data_dir):
    """
    Download the UCI EMG dataset from the official source and save it to the specified directory.

    URL: https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
    zip_path = data_dir / "uci_emg.zip"

    try:
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract and Flatten
        print("Extracting and cleaning up directory structure...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Zip usually has a top-level folder 'emg+data+for+gestures/'
                # We want to strip that and put contents directly in data_dir
                member_path = Path(member)
                
                # Skip the top-level directory itself and any non-essential files
                if len(member_path.parts) <= 1:
                    continue
                
                # Create the target path (omitting the first directory part)
                target_path = data_dir / Path(*member_path.parts[1:])
                
                if member.endswith('/'): # It's a directory
                    target_path.mkdir(parents=True, exist_ok=True)
                else: # It's a file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
        # Cleanup
        zip_path.unlink()
        print(f"Successfully downloaded and organized UCI-EMG in {data_dir}")
        
    except Exception as e:
        print(f"An error occurred during UCI-EMG download: {e}")
        if zip_path.exists():
            zip_path.unlink()



def convert_dataset(dataset_dir, output_fpath=None):
    with h5py.File(output_fpath, "w") as f:
        root = f.create_group("uci_emg")

        user_dirs = list(dataset_dir.iterdir())
        # User dir are numberic 
        user_dirs = [d for d in user_dirs if d.name.isdigit()]
        participant_num = 0
        for user_dir in tqdm(user_dirs):
            if not user_dir.is_dir(): continue

            participant_num += 1
            participant_id = f'sub-{participant_num:03d}'
            subj_group = root.create_group(participant_id)

            subj_group.attrs['participant_id'] = participant_id
            subj_group.attrs['origin_path'] = user_dir.name

            if participant_num in TRAIN_USERS:
                subj_split = "train"
            elif participant_num in VAL_USERS:
                subj_split = "val"
            elif participant_num in TEST_USERS:
                subj_split = "test"
            else:
                subj_split = "unknown"
                print(f"Warning: Participant {participant_id} (user_dir={user_dir.name}) not assigned to a known split. Marking as 'unknown'.")

            subj_group.attrs['inter_subject_split'] = subj_split

            # Assume both recordings are from the same session
            session_id = f'ses-001'
            session_group = subj_group.create_group(session_id)
            session_group.attrs['session_id'] = session_id

            run_num = 0
            gesture_counter = {}
            for trial_file in user_dir.glob("*.txt"):
                # Load the EMG data from the text file
                try:
                    # shapes (T,), (T, 8), (T,)
                    t, emg_data, labels = load_and_clean_uci_emg(trial_file)
                    # subtract 1 from labels so make -1 unkown and 0 rest
                    labels = labels - 1
                except Exception as e:
                    print(f"Error loading {trial_file}: {e}")
                    continue

                if not SEGMENT_GESTURES:
                    run_num += 1
                    run_id = f'run-{run_num:03d}'

                    run_group = session_group.create_group(run_id)
                    run_group.attrs['run_id'] = run_id
                    run_group.attrs['origin_path'] = f"{user_dir.name}/{trial_file.name}"
                    run_group.attrs['participant_id'] = participant_id
                    run_group.attrs['inter_subject_split'] = subj_split
                    run_group.attrs['inter_session_split'] = "n/a"  # All runs are from the same session
                    run_group.attrs['intra_session_split'] = "n/a"  # No segmentation into separate runs
                    run_group.attrs['sample_rate'] = 200

                    emg_data = emg_data / 0.0013  # Normalise emg to be between -1 and 1 (the original data is in

                    write_modality(run_group, "time", t, {'sample_rate': 200, 'units': 'seconds'})
                    write_modality(run_group, "emg", emg_data, {'sample_rate': 200, 'units': 'normalised [-1, 1]'})
                    write_modality(run_group, "labels", labels, {'sample_rate': 200, 'description': "Gesture class labels corresponding to each time point"})

                else:
                    times, emg_runs, labels, reps = segment_run(t, emg_data, labels, gesture_counter)
                    for t_run, emg_run, label_run, rep in zip(times, emg_runs, labels, reps):
                        run_num += 1
                        run_id = f'run-{run_num:03d}'

                        run_group = session_group.create_group(run_id)
                        run_group.attrs['run_id'] = run_id
                        run_group.attrs['origin_path'] = f"{user_dir.name}/{trial_file.name}"
                        run_group.attrs['participant_id'] = participant_id
                        run_group.attrs['inter_subject_split'] = subj_split
                        run_group.attrs['inter_session_split'] = "n/a"  # All runs are from the same session
                        run_group.attrs['sample_rate'] = 200

                        # Convert to appropriate dtypes for storage
                        t_run = np.array(t_run, dtype=np.float32)
                        emg_run = np.array(emg_run, dtype=np.float32)
                        # Normalise emg to be between -1 and 1 (the original data is in the range of -0.0013 to 0.0013
                        emg_run = emg_run / 0.0013
                        label_run = np.array(label_run, dtype=np.int16)

                        write_modality(run_group, "time", t_run, {'sample_rate': 200, 'units': 'seconds'})
                        write_modality(run_group, "emg", emg_run, {'sample_rate': 200, 'units': 'normalised [-1, 1]'})
                        write_modality(run_group, "labels", label_run, {'sample_rate': 200, 'description': "Gesture class labels corresponding to each time point"})

                        gesture_id = int(label_run[0])      # Assuming all labels in the run are the same gesture
                        run_group.attrs['gesture_id'] = gesture_id
                        run_group.attrs['is_labelled'] = gesture_id >= 0


                        if rep in TRAIN_REPS:
                            intra_session_split = "train"
                        elif rep in VAL_REPS:
                            intra_session_split = "val"
                        elif rep in TEST_REPS:
                            intra_session_split = "test"
                        else:
                            intra_session_split = "unknown"
                            print(f"Warning: Rep {rep} in {trial_file.name} not assigned to a known intra-session split. Marking as 'unknown'.")

                        run_group.attrs['intra_session_split'] = intra_session_split



def write_modality(run_group, modality_name, data_array, metadata):
    """Save a single modality dataset with its metadata to the run group."""
    run_group.create_dataset(modality_name, data=data_array, compression="lzf")
    for k, v in metadata.items():
        run_group[modality_name].attrs[k] = v
    
    if len(data_array.shape) > 1: 
        run_group[modality_name].attrs['n_channels'] = data_array.shape[1]

    run_group[modality_name].attrs['n_samples'] = data_array.shape[0]

    if 'sample_rate' not in metadata:
        print(f"Warning: Sample rate not provided for {modality_name} in {run_group.name}. Cannot calculate recording duration.")
    else:
        run_group[modality_name].attrs['recording_duration'] = data_array.shape[0] / metadata.get('sample_rate')




def segment_run(t, emg_data, labels, gesture_counter):
    diff = np.diff(labels, prepend=-2)
    run_starts = np.where(diff != 0)[0]
    run_ends = np.append(run_starts[1:], len(labels))

    times = []
    emg_runs = []
    label_runs = []
    reps = []

    for i, (start, end) in enumerate(zip(run_starts, run_ends)):
        gesture_id = labels[start]
        if gesture_id == -1 and not INCLUDE_UNKNOWN_LABELS:
            continue
        if gesture_id == 6 and not INCLUDE_EXTENDED_PALM:
            continue

        times.append(t[start:end])
        emg_runs.append(emg_data[start:end])
        label_runs.append(labels[start:end])

        gesture_counter[gesture_id] = gesture_counter.get(gesture_id, 0) + 1
        rep = gesture_counter[gesture_id]
        reps.append(rep)
    
    return times, emg_runs, label_runs, reps


def load_and_clean_uci_emg(file_path, target_fs=200):
    """
    UCI data has been oversampled in a wierd way and emg has a staircase effect. This function loads the raw text file, removes duplicate rows, and interpolates back to 200Hz

    Args:
        file_path (str or Path): Path to the raw text file containing EMG data.
        target_fs (int): Desired/original sampling frequency in Hz

    Returns:
        t_uniform (np.ndarray): Uniformly spaced timestamps in seconds. Shape (T,)
        emg_uniform (np.ndarray): EMG data interpolated to uniform timestamps. Shape (T, 8)
        labels_uniform (np.ndarray): Gesture class labels corresponding to each time point. Shape (T,)
    """
    # Load the raw text file
    # The files are space/tab separated with a header row
    df = pd.read_csv(file_path, sep='\t')
    
    # De-duplicate: Remove rows where EMG values are identical to previous row
    # We keep the first occurrence. This removes the 'staircase' effect.
    emg_cols = [c for c in df.columns if 'channel' in c.lower()]

    unique_mask = df[emg_cols].diff().abs().sum(axis=1) != 0
    df_unique = df[unique_mask].copy()
    
    # Convert 'time' column to seconds for the interpolator
    times = df_unique['time'].values / 1000.0 
    data = df_unique[emg_cols].values
    labels = df_unique['class'].values
    
    # We want a sample every 1/target_fs seconds (e.g., every 0.005s)
    t_start = times[0]
    t_end = times[-1]
    t_uniform = np.arange(t_start, t_end, 1/target_fs)
    
    # Interpolate EMG Signals
    # We use linear interpolation to estimate values at the exact 5ms marks
    f_emg = interp1d(times, data, axis=0, kind='linear', fill_value="extrapolate")
    emg_uniform = f_emg(t_uniform)
    
    # Interpolate Labels (Nearest Neighbor)
    # Labels shouldn't be averaged; we take the label of the closest original timestamp
    f_label = interp1d(times, labels, kind='nearest', fill_value="extrapolate")
    labels_uniform = f_label(t_uniform)
    
    return t_uniform, emg_uniform, labels_uniform


def main(dataset_dir=None, output_dir=None, download_data=False, include_rest=False, segment_gestures=True):
    hdf5_root = os.getenv("HDF5_ROOT")
    hdf5_root = Path(hdf5_root) if hdf5_root is not None else None

    if dataset_dir is None:
        # Default to HDF5_ROOT environment variable or current directory
        if hdf5_root is None:
            print("Warning: HDF5_ROOT environment variable not set. Please define the dataset directory or set HDF5_ROOT environment variable.")
            exit(1)
        print(f"HDF5_ROOT is set to: {hdf5_root}, dataset will be saved here")
        dataset_dir = hdf5_root / "neuromotor_wrist"
    
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        if hdf5_root is None:
            print("Warning: HDF5_ROOT environment variable not set and output_dir not provided. Output will be saved in the same directory as the input dataset.")
            output_fpath = dataset_dir / "neuromotor_wrist.hdf5"
        else:
            output_fpath = hdf5_root / "neuromotor_wrist" / "neuromotor_wrist.hdf5"
            output_fpath.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_fpath = output_dir / "neuromotor_wrist.hdf5"

    if download_data:
        download_dataset(dataset_dir)

    convert_dataset(dataset_dir, output_fpath)


def main(dataset_dir=None, output_dir=None, download_data=False, include_rest=False, segment_gestures=True, dataset_name="uci_emg"):
    hdf5_root = os.getenv("HDF5_ROOT")
    hdf5_root = Path(hdf5_root) if hdf5_root else None

    if dataset_dir is None:
        # Default to HDF5_ROOT environment variable or current directory
        if hdf5_root is None:
            print("Warning: HDF5_ROOT environment variable not set. Please define the dataset directory or set HDF5_ROOT environment variable.")
            exit(1)
        print(f"HDF5_ROOT is set to: {hdf5_root}, dataset will be saved here")
        dataset_dir = hdf5_root / dataset_name

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        if hdf5_root is None:
            print("Warning: HDF5_ROOT environment variable not set and output_dir not provided. Output will be saved in the same directory as the input dataset.")
            output_dir = dataset_dir
        else:
            output_dir = hdf5_root / dataset_name
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fpath = output_dir / f"{dataset_name}.hdf5"

    if download_data:
        download_dataset(dataset_dir)

    convert_dataset(dataset_dir, output_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/uci_emg/")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the converted HDF5 files. Defaults to the dataset directory.")
    parser.add_argument("--download_data", action="store_true", help="Whether to download the UCI-EMG dataset.")
    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir, args.download_data)