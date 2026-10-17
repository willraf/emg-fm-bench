""" Script ninapro_db5.py

Convert Ninapro DB5 from its original .mat format into a structured HDF5 format for efficient loading in our EMG processing pipeline.

NOTE:
    this script segments signals into individual gesture runs based on the 'restimulus' and 'rerepetition' labels.
    Rest portions (stimulus=0) are ignored in the final dataset, so final gesture IDs range from 1-52

Usage:
    % python -m scripts.datasets.ninapro_db5 --dataset_dir data/ninapro_db5/ --output_dir data/ninapro_db5/ --download_data --segment_gestures

Date:
    19/02/2026
"""

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
VAL_SUBJS = [7]
TEST_SUBJS = [5, 11]

TRAIN_REPS = [1, 3, 4, 6]
VAL_REPS = [2]
TEST_REPS = [5]

# Precomputed per-joint min/max from training subjects only (sub-01,03,04,06,08,09,10)
# Computed in notebooks/ninapro_db5_jntang_stats.ipynb
# Channel 10 has anomalously large range (~2000 units) — likely a sensor calibration issue
# Per-joint normalisation handles this correctly without special casing
JNTANG_TRAIN_MIN = np.array([
      -44.12763214,  -94.74728394,  -65.11074066,  -47.24693298,
      -64.37309265,  -28.87206841,  -18.53434753,  -49.31587982,
      -61.05163193,  -33.44902039, -1069.40991211,  -80.38005066,
      -38.44103241,  -38.09302902,  -23.54822731,  -36.09608459,
      -29.32806778,  -46.95530701,   -2.37881947,  -40.74927139,
     -109.05863953,  -19.48239899
], dtype=np.float32)

JNTANG_TRAIN_MAX = np.array([
     161.41166687,   75.17311859,   93.00727844,   70.04275513,
      96.21552277,  105.76824951,   86.50091553,  117.03561401,
     146.05380249,   47.89705276, 1076.46875,     129.52818298,
     123.36170197,   74.21311188,   16.05175972,  105.02424622,
     102.81341553,   84.05649567,   20.77408218,   66.91215515,
      63.17636108,   98.25905609
], dtype=np.float32)



def get_gesture_id(exercise_id, stimulus_id):
    """NinaPro DB5 has a unique gesture distribution across exercises:
    
    Each exercise has a different set of gestures:

    Exercise 1 (2 in fname, B in docs): 17 gestures (1-17)
    Exercise 2 (3 in fname, C in docs): 23 gestures (18-40)
    Exercise 3 (1 in fname, A in docs): 12 gestures (41-52)
    """

    if stimulus_id == 0: return 0 # Rest is always 0

    if exercise_id == 3:  # Exercise A
        return stimulus_id  # 1-12 map to 1-12
    elif exercise_id == 1:  # Exercise B
        return stimulus_id + 12  # 1-17 map to 13-29
    elif exercise_id == 2:  # Exercise C
        return stimulus_id + 12 + 17  # 1-23 map to 30-52
    else:
        raise ValueError(f"Unknown exercise_id {exercise_id} for gesture_id mapping")


def get_global_gesture_id(gesture_id):
    """Convert dataset-specific gesture IDs to a global standard for multi-dataset training
    currently left as placeholder"""
    return gesture_id


def exercise_id_to_name(exercise_id):
    if exercise_id == 3:
        return 'ExerciseA'
    elif exercise_id == 1:
        return 'ExerciseB'
    elif exercise_id == 2:
        return 'ExerciseC'
    else:
        return 'Unkown'

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

    subject_num = int(mat['subject'][0][0])
    # show as two numeric digits for better sorting (e.g. sub-01, sub-02, ..., sub-10)
    subject_id = f"sub-{subject_num:02d}"

    standard_user = {
        'participant_id': subject_id,
        'age': int(mat['age'][0][0]),
        'sex': gender,
        'handedness': handedness,
        'weight': float(mat['weight'][0][0]),
        'height': float(mat['height'][0][0]),
        'forearm_circumference': float(mat['circumference'][0][0]),
    }
    
    standard_session = {
        'inter_session_split': 'n/a',  # NinaPro DB5 doesn't have multiple sessions per subject
        'side': 'r',  # All right hand movements in NinaPro DB5
    }

    standard_run = {
        'exercise_id': int(mat['exercise'][0][0]),
        'sample_rate': int(mat['frequency'][0][0]),
        'task_name': exercise_id_to_name(int(mat['exercise'][0][0])),
        'side': 'r',  # All right hand movements in NinaPro DB5
    }
    
    return standard_user, standard_session, standard_run

def download_dataset(data_dir):
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


def write_modality(run_group, modality_name, data_array, metadata):
    run_group.create_dataset(modality_name, data=data_array, compression="lzf")
    for k, v in metadata.items():
        run_group[modality_name].attrs[k] = v
    
    if len(data_array.shape) > 1: 
        run_group[modality_name].attrs['n_channels'] = data_array.shape[1]
    run_group[modality_name].attrs['n_samples'] = data_array.shape[0]
    run_group[modality_name].attrs['recording_duration'] = data_array.shape[0] / metadata.get('sample_rate', 200)


def convert_dataset(dataset_dir, output_fpath=None, include_rest=False, segment_gestures=True, normalise_joint_angles=True):

    # Should work with both nested and flat dir structures
    mat_files = sorted(dataset_dir.rglob("*.mat"))

    with h5py.File(output_fpath, 'w') as f:
        root = f.create_group("ninapro_db5")
        root.attrs['jntang_train_min'] = JNTANG_TRAIN_MIN
        root.attrs['jntang_train_max'] = JNTANG_TRAIN_MAX        

        for fpath in tqdm(mat_files, desc="Converting NinaPro DB5"):
            mat = scipy.io.loadmat(str(fpath))

            user_std, sess_std, run_std = process_metadata(mat)
            
            # --------------------------------------------------
            #            Create or open subject group
            # --------------------------------------------------
            subj_key = user_std['participant_id']
            if subj_key not in root:
                subj_group = root.create_group(subj_key)
                for k, v in user_std.items():
                    subj_group.attrs[k] = v

                subj_group.attrs['origin_path'] = str(fpath)

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


            # --------------------------------------------------
            #            Create or open session group
            # --------------------------------------------------
            # We treat the three exercises as a single session per subject, as they do not explicitly state that the sensors were removed/reapplied between exercises
            session_key = f"ses-01"
            sess_std['session_id'] = session_key
            if session_key not in subj_group:
                sess_group = subj_group.create_group(session_key)
                for k, v in sess_std.items():
                    sess_group.attrs[k] = v

                sess_group.attrs['origin_path'] = str(fpath)
            else:
                sess_group = subj_group[session_key]


            # --------------------------------------------------
            #            get timeseries data and labels
            # --------------------------------------------------
            # Continuous data arrays
            emg = mat['emg'].astype(np.float32) / 128.0 # Standardize to [-1, 1]
            acc = mat['acc'].astype(np.float32)
            glove = mat['glove'].astype(np.float32)

            if normalise_joint_angles:
                # Normalise joint angles to [0, 1] based on training set min/max
                glove = (glove - JNTANG_TRAIN_MIN) / (JNTANG_TRAIN_MAX - JNTANG_TRAIN_MIN)
                glove = np.clip(glove, 0.0, 1.0)
            
            # Use 'restimulus' and 'rerepetition' for refined labels
            stimulus = mat['restimulus'].flatten()
            repetition = mat['rerepetition'].flatten()


            # Include the full unsegmented data as a "run"
            if not segment_gestures:
                exercise_id = run_std['exercise_id']
                run_key = f"run-{exercise_id:02d}"
                run_std['run_id'] = run_key

                run_group = sess_group.create_group(run_key)
                run_std['intra_session_split'] = 'n/a'   # Without segmentation, we can't assign train/val/test splits at the run level based on repetitions

                for k, v in run_std.items():
                    run_group.attrs[k] = v

                run_group.attrs['origin_path'] = str(fpath)

                # NinaPro DB5 has already been resampled to 200Hz, so all modalities should be aligned by index
                emg_meta = {
                    'sample_rate': run_std.get('sample_rate', 200),
                    'sensor_type': str(mat.get('sensor', ['unknown'])[0]),
                    'units': 'normalized [-1, 1]',
                }
                write_modality(run_group, "emg", emg, emg_meta)

                accel_meta = {
                    'sample_rate': run_std.get('sample_rate', 200),
                    'sensor_type': str(mat.get('sensor', ['unknown'])[0]),
                }
                write_modality(run_group, "accel", acc, accel_meta)

                jntang_meta = {
                    'sample_rate': run_std.get('sample_rate', 200),
                    'sensor_type': 'CyberGlove II',
                }
                if normalise_joint_angles:
                    jntang_meta['units'] = 'normalised [0, 1]'
                else: 
                    jntang_meta['units'] = 'raw'

                write_modality(run_group, "jntang", glove, jntang_meta)

                gesture_id_meta = {
                    'description': "Original restimulus labels from NinaPro DB5",
                    'sample_rate': run_std.get('sample_rate', 200),
                }
                gesture_id_array = np.array([get_gesture_id(exercise_id, stim) for stim in stimulus], dtype=np.int16)
                write_modality(run_group, "gesture_id", gesture_id_array, gesture_id_meta)

                repetition_meta = {
                    'description': "Original rerepetition labels from NinaPro DB5",
                    'sample_rate': run_std.get('sample_rate', 200),
                }
                write_modality(run_group, "repetition", repetition, repetition_meta)

                global_gesture_meta = {
                    'description': "Global gesture IDs mapped from original restimulus labels, for cross-dataset consistency",
                    'sample_rate': run_std.get('sample_rate', 200),
                }
                global_gesture_ids = np.array([get_global_gesture_id(get_gesture_id(run_std['exercise_id'], stim)) for stim in stimulus], dtype=np.int16)
                write_modality(run_group, "global_gesture_id", global_gesture_ids, global_gesture_meta)

            else:    

                # Identify transitions to catch "Rest" periods
                # A "run" is now defined as any continuous block with the same stimulus ID
                diff = np.diff(stimulus, prepend=-1)
                run_starts = np.where(diff != 0)[0]
                run_ends = np.append(run_starts[1:], len(stimulus))

                for i, (start, end) in enumerate(zip(run_starts, run_ends)):
                    stim_idx = stimulus[start]
                    rep_idx = repetition[start]
                    # Convert from int8 to int to prevent overflow
                    stim_idx = int(stim_idx)
                    rep_idx = int(rep_idx)

                    if not include_rest and stim_idx == 0:
                        continue
                    
                    # We still ignore very short windows
                    # NOTE: including rest is likely to cause label leakage, as distribution of signal lengths is very different for rest vs active periods.
                    if (end - start) < 200: 
                        continue
                    
                    exercise_id = run_std['exercise_id']
                    gesture_id = get_gesture_id(exercise_id, stim_idx)

                    # 3. Handle naming for Rest (Label 0)
                    # We add a unique counter 'T' for run because Repetition 0 isn't unique
                    if include_rest:
                        new_run_id = int(rep_idx) + (6 * (gesture_id))
                    else:
                        new_run_id = int(rep_idx) + (6 * (gesture_id-1))  # Shift down by 1 to account for excluded rest

                    run_key = f"run-{new_run_id:03d}"  # Zero-pad for consistent sorting
                    run_group = sess_group.create_group(run_key)
                    
                    # Metadata
                    run_std['gesture_id'] = gesture_id
                    run_std['run_id'] = run_key
                    run_std['rep'] = rep_idx

                    if rep_idx == 2:  # Validation repetition
                        run_std['intra_session_split'] = 'val'
                    elif rep_idx == 5:  # Test repetition
                        run_std['intra_session_split'] = 'test'
                    else:
                        run_std['intra_session_split'] = 'train'

                    for k, v in run_std.items():
                        run_group.attrs[k] = v
                    
                    run_group.attrs['origin_path'] = str(fpath)

                    # All modalities are oversampled to 200Hz, so they should be aligned by index
                    emg_meta = {
                        'sample_rate': run_std.get('sample_rate', 200),
                        'sensor_type': str(mat.get('sensor', ['unknown'])[0]),
                        'units': 'normalized [-1, 1]',
                    }
                    write_modality(run_group, "emg", emg[start:end], emg_meta)

                    accel_meta = {
                        'sample_rate': run_std.get('sample_rate', 200),
                        'sensor_type': str(mat.get('sensor', ['unknown'])[0]),
                    }
                    write_modality(run_group, "accel", acc[start:end], accel_meta)

                    jntang_meta = {
                        'sample_rate': run_std.get('sample_rate', 200),
                        'sensor_type': 'CyberGlove II',
                    }
                    if normalise_joint_angles:
                        jntang_meta['units'] = 'normalised [0, 1]'
                    else: 
                        jntang_meta['units'] = 'raw'
                    write_modality(run_group, "jntang", glove[start:end], jntang_meta)


def main(dataset_dir=None, output_dir=None, download_data=False, include_rest=False, segment_gestures=True, dataset_name="ninapro_db5"):
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

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output HDF5 file will be saved to: {output_fpath}")

    if download_data:
        download_dataset(dataset_dir)

    convert_dataset(dataset_dir, output_fpath, include_rest=include_rest, segment_gestures=segment_gestures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory where the raw .mat files are located (or will be downloaded to if --download_data is set)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the converted HDF5 file. Defaults to the HDF5 root if set, then dataset directory.")
    parser.add_argument("--download_data", action='store_true', help="If set, will attempt to download the raw .mat files from the official Ninapro DB5 source")
    parser.add_argument("--include_rest", action='store_true', help="If set, will include rest periods (stimulus=0) as a separate class in the dataset. WARNING: This may cause label leakage due to different signal lengths and distributions for rest vs active periods.")
    parser.add_argument("--segment_gestures", action='store_true', help="If set, will segment continuous recordings into individual gesture runs based on stimulus transitions. Otherwise, each exercise will be treated as a single run.")
    args = parser.parse_args()
    
    main(args.dataset_dir, args.output_dir, args.download_data, include_rest=args.include_rest, segment_gestures=args.segment_gestures)