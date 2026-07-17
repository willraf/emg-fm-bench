""" Script ninapro_db2.py

Convert Ninapro DB2 from its original .mat format into a structured HDF5 format for efficient loading in our EMG processing pipeline.

NOTE:
    this script segments signals into individual gesture runs based on the 'restimulus' and 'rerepetition' labels.
    Rest portions (stimulus=0) are ignored in the final dataset, so final gesture IDs range from 1-52

Usage:
    % python -m scripts.datasets.ninapro_db2 --dataset_dir data/ninapro_db2/ --output_dir data/ninapro_db2/ --download_data

Authors:
    Will Raftery

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

TRAIN_SUBJS = [2, 3, 4, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 36, 37, 39, 40]
VAL_SUBJS = [1, 13, 31, 33]
TEST_SUBJS = [5, 11, 14, 16, 20, 21, 34, 38]

# These statistics are computed over the training set only, and are used to normalise the force values
# See notebooks/ninapro_db2.ipynb for code
FORCE_TRAIN_MIN = np.array([-4.980, -4.686, -8.786, -8.105, -15.814, -8.643], dtype=np.float32)
FORCE_TRAIN_MAX = np.array([27.539, 28.613, 41.016, 49.509,  39.740, 65.722], dtype=np.float32)



SUBJ_METADATA = {
    1:  {'sex': 'm', 'handedness': 'r', 'age': 29, 'height': 187, 'weight': 75},
    2:  {'sex': 'm', 'handedness': 'r', 'age': 29, 'height': 183, 'weight': 75},
    3:  {'sex': 'm', 'handedness': 'r', 'age': 31, 'height': 174, 'weight': 69},
    4:  {'sex': 'f', 'handedness': 'l', 'age': 30, 'height': 154, 'weight': 50},
    5:  {'sex': 'm', 'handedness': 'r', 'age': 25, 'height': 175, 'weight': 70},
    6:  {'sex': 'm', 'handedness': 'r', 'age': 35, 'height': 172, 'weight': 79},
    7:  {'sex': 'm', 'handedness': 'r', 'age': 27, 'height': 187, 'weight': 92},
    8:  {'sex': 'm', 'handedness': 'r', 'age': 45, 'height': 173, 'weight': 73},
    9:  {'sex': 'm', 'handedness': 'r', 'age': 23, 'height': 172, 'weight': 63},
    10: {'sex': 'm', 'handedness': 'r', 'age': 34, 'height': 173, 'weight': 84},
    11: {'sex': 'f', 'handedness': 'r', 'age': 32, 'height': 150, 'weight': 54},
    12: {'sex': 'm', 'handedness': 'r', 'age': 29, 'height': 184, 'weight': 90},
    13: {'sex': 'm', 'handedness': 'l', 'age': 30, 'height': 182, 'weight': 70},
    14: {'sex': 'f', 'handedness': 'r', 'age': 30, 'height': 173, 'weight': 59},
    15: {'sex': 'm', 'handedness': 'r', 'age': 30, 'height': 169, 'weight': 58},
    16: {'sex': 'm', 'handedness': 'r', 'age': 34, 'height': 173, 'weight': 76},
    17: {'sex': 'm', 'handedness': 'r', 'age': 29, 'height': 175, 'weight': 70},
    18: {'sex': 'f', 'handedness': 'r', 'age': 30, 'height': 169, 'weight': 90},
    19: {'sex': 'f', 'handedness': 'r', 'age': 31, 'height': 158, 'weight': 52},
    20: {'sex': 'f', 'handedness': 'r', 'age': 26, 'height': 155, 'weight': 52},
    21: {'sex': 'm', 'handedness': 'r', 'age': 32, 'height': 170, 'weight': 75},
    22: {'sex': 'f', 'handedness': 'l', 'age': 28, 'height': 162, 'weight': 54},
    23: {'sex': 'm', 'handedness': 'r', 'age': 25, 'height': 170, 'weight': 66},
    24: {'sex': 'm', 'handedness': 'r', 'age': 28, 'height': 170, 'weight': 73},
    25: {'sex': 'm', 'handedness': 'l', 'age': 31, 'height': 168, 'weight': 70},
    26: {'sex': 'm', 'handedness': 'l', 'age': 30, 'height': 186, 'weight': 90},
    27: {'sex': 'm', 'handedness': 'r', 'age': 29, 'height': 170, 'weight': 65},
    28: {'sex': 'f', 'handedness': 'r', 'age': 29, 'height': 160, 'weight': 61},
    29: {'sex': 'm', 'handedness': 'r', 'age': 27, 'height': 171, 'weight': 64},
    30: {'sex': 'm', 'handedness': 'r', 'age': 30, 'height': 173, 'weight': 68},
    31: {'sex': 'm', 'handedness': 'r', 'age': 29, 'height': 185, 'weight': 98},
    32: {'sex': 'm', 'handedness': 'r', 'age': 28, 'height': 173, 'weight': 72},
    33: {'sex': 'm', 'handedness': 'r', 'age': 25, 'height': 183, 'weight': 71},
    34: {'sex': 'm', 'handedness': 'r', 'age': 31, 'height': 192, 'weight': 78},
    35: {'sex': 'f', 'handedness': 'r', 'age': 24, 'height': 170, 'weight': 52},
    36: {'sex': 'f', 'handedness': 'r', 'age': 27, 'height': 155, 'weight': 44},
    37: {'sex': 'm', 'handedness': 'r', 'age': 34, 'height': 190, 'weight': 105},
    38: {'sex': 'f', 'handedness': 'r', 'age': 30, 'height': 163, 'weight': 62},
    39: {'sex': 'm', 'handedness': 'r', 'age': 31, 'height': 183, 'weight': 96},
    40: {'sex': 'm', 'handedness': 'r', 'age': 31, 'height': 173, 'weight': 65},
}


def get_gesture_id(exercise_id, stimulus_id):
    """NinaPro db2 has a unique gesture distribution across exercises.
    Unlike the db5 dataset, gesture IDs in db2 are unique across exercises,
      so we just need to shift them to account for fact that exercise A (gestures 1-12) is not present in db2.
    
    Each exercise has a different set of gestures:

    Exercise 1 (1 in fname, B in docs): 17 gestures (1-17)
    Exercise 2 (2 in fname, C in docs): 23 gestures (18-40)
    Exercise 3 (3 in fname, D in docs): 9 gestures (41-49)
    Rest (0 in stimulus) is common across all exercises and will be mapped to 0
    """

    if stimulus_id == 0: return 0 # Rest is always 0

    return stimulus_id + 12


def get_global_gesture_id(gesture_id):
    """Convert dataset-specific gesture IDs to a global standard for multi-dataset training
    currently left as placeholder"""
    return gesture_id


def exercise_id_to_name(exercise_id):
    if exercise_id == 1:
        return 'ExerciseB'
    elif exercise_id == 2:
        return 'ExerciseC'
    elif exercise_id == 3:
        return 'ExerciseD'
    else:
        return 'Unknown'


def process_metadata(mat):
    """Standardize NinaPro metadata to match DATA_SPEC.md"""

    subject_id = int(mat['subject'][0][0])
    assert subject_id in SUBJ_METADATA, f"Subject ID {subject_id} not found in predefined metadata dictionary. Please update SUBJ_METADATA with this subject's info."

    subject_metadata = SUBJ_METADATA[subject_id]
    subject_metadata['participant_id'] = f"sub-{subject_id:02d}"  # Standardize to sub-XX format

    
    session_metadata = {
        'inter_session_split': 'n/a',  # NinaPro DB2 doesn't have multiple sessions per subject
        'side': 'r',  # All right hand movements in NinaPro db2
    }

    run_metadata = {
        'exercise_id': int(mat['exercise'][0][0]),
        'task_name': exercise_id_to_name(int(mat['exercise'][0][0])),
        'side': 'r',  # All right hand movements in NinaPro db2
    }
    
    return subject_metadata, session_metadata, run_metadata

def download_dataset(data_dir):
    """
    Downloads and flattens the NinaPro DB2 structure.
    Official URL: https://ninapro.hevs.ch/files/DB2_Preproc/
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://ninapro.hevs.ch/files/DB2_Preproc/"
    
    for i in range(1, 41):  # Subjects 1-40
        url = f"{base_url}DB2_s{i}.zip"
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
    run_group[modality_name].attrs['recording_duration'] = data_array.shape[0] / metadata.get('sample_rate')


def convert_dataset(dataset_dir, output_fpath=None, include_rest=False, segment_gestures=True, normalise_force=False):

    # Should work with both nested and flat dir structures
    mat_files = sorted(dataset_dir.rglob("*.mat"))

    with h5py.File(output_fpath, 'w') as f:
        root = f.create_group("ninapro_db2")

        root.attrs['force_train_min'] = FORCE_TRAIN_MIN
        root.attrs['force_train_max'] = FORCE_TRAIN_MAX
        
        for fpath in tqdm(mat_files, desc="Converting NinaPro DB2"):
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
            emg = mat['emg'].astype(np.float32)
            acc = mat['acc'].astype(np.float32)

            glove = mat['glove'].astype(np.float32) if 'glove' in mat else None
            inclin = mat['inclin'].astype(np.float32) if 'inclin' in mat else None
            force = mat['force'].astype(np.float32) if 'force' in mat else None

            if force is not None and normalise_force:
                force = (force - FORCE_TRAIN_MIN) / (FORCE_TRAIN_MAX - FORCE_TRAIN_MIN + 1e-6)
                force = np.clip(force, 0.0, 1.0)    

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

                # NinaPro db2 has already been resampled to 200Hz, so all modalities should be aligned by index
                emg_meta = {
                    'sample_rate': 2000,
                    'sensor_type': 'Delsys Trigno Wireless system',
                    'units': 'V',
                }
                write_modality(run_group, "emg", emg, emg_meta)

                accel_meta = {
                    'sample_rate': 2000,
                    'sensor_type': 'Built-in Delsys Trigno accelerometers'
                }
                write_modality(run_group, "accel", acc, accel_meta)

                if glove is not None: 
                    jntang_meta = {
                        'sample_rate': 2000,
                        'sensor_type': 'CyberGlove II',
                        'units': 'uncalibrated raw sensor values',
                    }
                    write_modality(run_group, "jntang", glove, jntang_meta)

                if inclin is not None:
                    inclin_meta = {
                        'sample_rate': 2000,
                        'sensor_type': 'IS40 inclinometer',
                        'units': 'degrees',
                    }
                    write_modality(run_group, "inclin", inclin, inclin_meta)

                if force is not None:
                    force_meta = {
                        'sample_rate': 2000,
                        'sensor_type': 'FSR force sensors',
                        'units': '% of MVC',
                    }
                    write_modality(run_group, "force", force, force_meta)

                gesture_id_meta = {
                    'description': "Original restimulus labels from NinaPro db2",
                    'sample_rate': 2000,
                }
                gesture_id_array = np.array([get_gesture_id(exercise_id, stim) for stim in stimulus], dtype=np.int16)
                write_modality(run_group, "gesture_id", gesture_id_array, gesture_id_meta)

                repetition_meta = {
                    'description': "Original rerepetition labels from NinaPro db2",
                    'sample_rate': 2000,
                }
                write_modality(run_group, "repetition", repetition, repetition_meta)

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
                    if (end - start) < 1000: 
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
                        'sample_rate': 2000,
                        'sensor_type': 'Delsys Trigno Wireless system',
                        'units': 'V',
                    }
                    write_modality(run_group, "emg", emg[start:end], emg_meta)

                    accel_meta = {
                        'sample_rate': 2000,
                        'sensor_type': 'Built-in Delsys Trigno accelerometers'
                    }
                    write_modality(run_group, "accel", acc[start:end], accel_meta)

                    if glove is not None: 
                        jntang_meta = {
                            'sample_rate': 2000,
                            'sensor_type': 'CyberGlove II',
                            'units': 'uncalibrated raw sensor values',
                        }
                        write_modality(run_group, "jntang", glove[start:end], jntang_meta)

                    if inclin is not None:
                        inclin_meta = {
                            'sample_rate': 2000,
                            'sensor_type': 'IS40 inclinometer',
                            'units': 'degrees',
                        }
                        write_modality(run_group, "inclin", inclin[start:end], inclin_meta)

                    if force is not None:
                        force_meta = {
                            'sample_rate': 2000,
                            'sensor_type': 'FSR force sensors',
                            'units': '% of MVC',
                        }
                        write_modality(run_group, "force", force[start:end], force_meta)



def main(dataset_dir=None, output_dir=None, download_data=False, include_rest=False, segment_gestures=True, dataset_name="ninapro_db2"):
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
    parser.add_argument("--download_data", action='store_true', help="If set, will attempt to download the raw .mat files from the official Ninapro DB2 source")
    parser.add_argument("--include_rest", action='store_true', help="If set, will include rest periods (stimulus=0) as a separate class in the dataset. WARNING: This may cause label leakage due to different signal lengths and distributions for rest vs active periods.")
    parser.add_argument("--segment_gestures", action='store_true', help="If set, will segment continuous recordings into individual gesture runs based on stimulus transitions. Otherwise, each exercise will be treated as a single run.")
    args = parser.parse_args()
    
    main(args.dataset_dir, args.output_dir, args.download_data, include_rest=args.include_rest, segment_gestures=args.segment_gestures)