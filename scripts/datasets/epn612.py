""" Script epn.py

Convert epn612 dataset from JSON to HDF5 format. Optionally, download the raw dataset from Zenodo if not already available locally.

Usage:
    % python -m scripts.datasets.epn --dataset_dir data/epn612/ --output_dir data/epn612/ --download_data --segment_gestures


Authors:
    Will Raftery

Date:
    14/03/2026
"""


import os

import h5py
import json
import numpy as np
import argparse
import urllib.request
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

TRAIN_REPS = [1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 16, 17, 19, 20, 21, 23, 24, 25]
VAL_REPS = [7, 15]
TEST_REPS = [3, 8, 12, 18, 22]

assert len(set(TRAIN_REPS) & set(VAL_REPS)) == 0, "Overlap between train and val reps"
assert len(set(TRAIN_REPS) & set(TEST_REPS)) == 0, "Overlap between train and test reps"
assert len(set(VAL_REPS) & set(TEST_REPS)) == 0, "Overlap between val and test reps"
assert len(set(TRAIN_REPS) | set(VAL_REPS) | set(TEST_REPS)) == 25, "Total unique reps should be 25"

def get_gesture_id(gesture_name):
    """Maps a gesture name to its corresponding ID using the provided mapping."""

    gesture_mapping = {
        "noGesture": 0,
        "fist": 1,
        "waveIn": 2,
        "waveOut": 3,
        "open": 4,
        "pinch": 5
    }

    return gesture_mapping.get(gesture_name, -1)  # Return -1 for unknown gestures


def get_global_gesture_id(gesture_id):
    """Maps local (datset specific) gesture IDs to global for multi-dataset training"""
    return gesture_id

def process_metadata(user_info, general_info, participant_id, json_file):
    gender = user_info.get('gender', 'u').lower()

    if gender in ['man', 'male']:
        gender = 'm'
    elif gender in ['woman', 'female']:
        gender = 'f'
    elif gender in ['other', 'non-binary', 'nonbinary']:
        gender = 'o'
    else:
        gender = 'u'

    handedness = user_info.get('handedness', 'u').lower()
    if handedness in ['right', 'r']:
        handedness = 'r'
    elif handedness in ['left', 'l']:
        handedness = 'l'
    else:
        handedness = 'u'

    standardised_user_info = {
        'participant_id': participant_id,
        'age': int(user_info.get('age', -1)),
        'sex': gender,
        'handedness': handedness,
        'occupation': user_info.get('occupation', 'unknown'),
        'ethnicity': user_info.get('ethnicGroup', 'unknown'),
        'arm_damage': user_info.get('ArmDamage', 'unknown'),
        'distance_elbow_to_myo_cm': float(user_info.get('distanceFromElbowToMyoInCm', -1)),
        'distance_elbow_to_ulna_cm': float(user_info.get('distanceFromElbowToUlnaInCm', -1)),
        'forearm_circumference': float(user_info.get('armPerimeterInCm', -1)),
        # Store the original JSON file path for traceability
        'origin_path': str(json_file)
    }

    standardised_session_info = {
        'session_id': 'ses-001',  # EPN612 does not explicitly state that each sample (idx_1, idx_2, etc.) is from a different recording session
        'device_model': general_info.get('deviceModel', 'unknown'),
        'sample_rate': int(general_info.get('samplingFrequencyInHertz', -1)),
        'recording_time_s': int(general_info.get('recordingTimeInSeconds', -1)),
        'repetitions_for_sync_gesture': int(general_info.get('repetitionsForSynchronizationGesture', -1)),
        'date': user_info.get('date', 'unknown'),
        'side': 'r',
        'inter_session_split': 'n/a',  
        'origin_path': str(json_file)
    }

    standardised_run_info = {
        'sample_rate': int(general_info.get('samplingFrequencyInHertz', -1)),
    }

    return standardised_user_info, standardised_session_info, standardised_run_info


def download_dataset(data_dir):
    """
    Downloads EPN612 from Zenodo and flattens the directory structure.
    URL: https://zenodo.org/records/4421500
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://zenodo.org/records/4421500/files/EMG-EPN612%20Dataset.zip?download=1"
    zip_path = data_dir / "epn612_raw.zip"
    
    print(f"Downloading EPN612 from Zenodo...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        
        # 2. Extract and Flatten
        print("Extracting and cleaning up directory structure...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Zenodo zip usually has a top-level folder 'EMG-EPN612 Dataset/'
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
                        
        # 3. Cleanup
        zip_path.unlink()
        print(f"Successfully downloaded and organized EPN612 in {data_dir}")
        
    except Exception as e:
        print(f"An error occurred during EPN612 download: {e}")
        if zip_path.exists():
            zip_path.unlink()


def process_run(run_std, run_group, run_json_data, origin_path, new_run_id, gesture_counter, segment_gesture_flag=False):
    """
    Process a single session data and store it in the session_group.
    """
    
    # Copy session specific metadata that was included under user
    run_metadata = run_std.copy()
    
    # Determine intra-session split based on gesture repetition 
    gesture_name = run_json_data.get('gestureName', 'unknown')
    gesture_id = get_gesture_id(gesture_name)
    gesture_counter[gesture_id] = gesture_counter.get(gesture_id, 0) + 1
    rep = gesture_counter[gesture_id]

    if gesture_id < 0:
        session_split = "unlabelled"
        run_metadata['is_labelled'] = False         # Mark as unlabelled if gesture is unknowm for pytroch dataset to ignore
        # If the there is no gesture label, we mark the split as unlabelled as the sample cannot be used for finetuning or evaluation. 
    else:
        run_metadata['is_labelled'] = True          # Mark as labelled if gesture is known for pytroch dataset to use for finetuning and evaluation

        # Assign splits based on rep number
        if rep in TRAIN_REPS:
            session_split = "train"
        elif rep in VAL_REPS:   
            session_split = "val"
        elif rep in TEST_REPS:
            session_split = "test"
        else:
            print(f"Warning: Gesture {gesture_name} (ID {gesture_id}) repetition {rep} does not fit into any split. Marking as unlabelled.")
            session_split = "unlabelled"

    run_metadata['intra_session_split'] = session_split

    # Add session metadata to session_std
    run_metadata['gesture_name'] = gesture_name
    run_metadata['gesture_id'] = gesture_id
    run_metadata['start_point'] = run_json_data.get('startPointforGestureExecution', 0)
    run_metadata['run_id'] = new_run_id
    run_metadata['origin_path'] = origin_path
    run_metadata['rep'] = rep

    for k, v in run_metadata.items():
        run_group.attrs[k] = v

    # Loads as dictionary of channel names to lists of values, e.g. 'ch1': [0.1, 0.2, ...]
    emg_dict = run_json_data.get('emg', {})
    # Shape (8, T)
    emg_data = np.array([emg_dict.get(f'ch{i}', []) for i in range(1, 9)], dtype=np.float32)

    # Dead sample check
    if emg_data.size == 0 or np.std(emg_data) == 0:
        return

    sample_rate = run_metadata.get('sample_rate', 200)

    if segment_gesture_flag:
        window_len = 1.0
        if gesture_id == 0:
            # For rest/noGesture, take the middle window
            mid = emg_data.shape[1] // 2
            half = int(window_len * sample_rate // 2)
            start, end = mid - half, mid + half
            start = max(0, start)
            end   = min(emg_data.shape[1], end)
        else:
            start, end = segment_gesture(emg_data, sample_rate, window_s=window_len)
    else:
        start, end = 0, emg_data.shape[1]
    
    run_group.attrs['segment_start'] = start
    run_group.attrs['segment_end']   = end


    emg_seg = emg_data[:, start:end]   # (8, T')
    emg_seg = emg_seg / 128.0          # normalise to [-1, 1]
    emg_seg = emg_seg.T                # (T', 8)

    write_modality(run_group, "emg", emg_seg, {
        'sample_rate': sample_rate,
        'sensor_type': 'Thalmic Myo Armband',
        'units':       'normalized [-1, 1]',
    })

    imu_sample_rate = 50
    imu_start = round(start * (imu_sample_rate / sample_rate))
    imu_end   = round(end   * (imu_sample_rate / sample_rate))

    accel = run_json_data.get('accelerometer', {})
    accel_data = np.array([accel.get(axis, []) for axis in ['x', 'y', 'z']], dtype=np.float32).T  # Shape (T, 3)
    accel_data = accel_data[imu_start:imu_end, :]  # Align
    accel_meta = {
        'sample_rate': imu_sample_rate,
        'sensor_type': 'Thalmic Myo Armband',
    }
    write_modality(run_group, "accel", accel_data, accel_meta)

    gyro = run_json_data.get('gyroscope', {})
    gyro_data = np.array([gyro.get(axis, []) for axis in ['x', 'y', 'z']], dtype=np.float32).T  # Shape (T, 3)
    gyro_data = gyro_data[imu_start:imu_end, :]  # Align
    gyro_meta = {
        'sample_rate': imu_sample_rate,
        'sensor_type': 'Thalmic Myo Armband',
    }
    write_modality(run_group, "gyro", gyro_data, gyro_meta)

    quat = run_json_data.get('quaternion', {})
    quat_data = np.array([quat.get(axis, []) for axis in ['w', 'x', 'y', 'z']], dtype=np.float32).T  # Shape (T, 4)
    quat_data = quat_data[imu_start:imu_end, :]  # Align
    quat_meta = {
        'sample_rate': imu_sample_rate,
        'sensor_type': 'Thalmic Myo Armband',
    }
    write_modality(run_group, "quat", quat_data, quat_meta)



    # # 3. IMU Dataset (Structured)
    # DEPRICATED, no longer using compound datasets
    # imu_dtype = np.dtype([
    #     ('accelerometer', '<f4', (3,)),
    #     ('gyroscope', '<f4', (3,)),
    #     ('quaternion', '<f4', (4,))
    # ])

    # T_imu = min(len(accel.get('x', [])), len(gyro.get('x', [])), len(quat.get('w', [])))
    # if T_imu == 0: continue

    # imu_array = np.empty(T_imu, dtype=imu_dtype)
    # for i, axis in enumerate(['x', 'y', 'z']):
    #     imu_array['accelerometer'][:, i] = accel.get(axis, [])[:T_imu]
    #     imu_array['gyroscope'][:, i] = gyro.get(axis, [])[:T_imu]
    # for i, axis in enumerate(['w', 'x', 'y', 'z']):
    #     imu_array['quaternion'][:, i] = quat.get(axis, [])[:T_imu]

    # session_group.create_dataset("imu", data=imu_array, compression="gzip")


def segment_gesture(emg_data: np.ndarray,
                    sample_rate: int,
                    window_s: float = 2.0,
                    smooth_ms: int = 100) -> tuple[int, int]:
    """
    Find the most active window in an EMG recording using a smoothed
    RMS envelope, then return the start and end sample indices of the
    best window.

    Strategy:
        1. Compute per-sample RMS across all channels
        2. Smooth with a rectangular (boxcar) window to get a stable envelope
        3. Use a sliding window of length `window_s` seconds to find the
           position with the highest mean envelope energy
        4. Return that window's start and end indices

    For 'noGesture' recordings the signal is approximately uniform, so
    the function still returns a valid window (the highest-energy region,
    which is arbitrary but consistent).

    Args:
        emg_data:   Raw EMG array of shape (C, T) — channels first, BEFORE
                    normalisation and transpose
        sample_rate: Sampling frequency in Hz
        window_s:   Desired output window length in seconds
        smooth_ms:  Smoothing kernel width in milliseconds

    Returns:
        (start, end) sample indices into the original T dimension
    """
    C, T = emg_data.shape
    window_samples = int(window_s * sample_rate)
    smooth_samples = max(1, int(smooth_ms * sample_rate / 1000))

    # ------------------------------------------------------------------
    # 1. RMS across channels → shape (T,)
    # ------------------------------------------------------------------
    rms = np.sqrt(np.mean(emg_data ** 2, axis=0))   # (T,)

    # ------------------------------------------------------------------
    # 2. Smooth with a boxcar (uniform) kernel
    # ------------------------------------------------------------------
    kernel = np.ones(smooth_samples) / smooth_samples
    rms_smooth = np.convolve(rms, kernel, mode='same')   # (T,)

    # ------------------------------------------------------------------
    # 3. Sliding window sum to find highest-energy position
    #    Use cumsum trick for O(T) efficiency
    # ------------------------------------------------------------------
    if T <= window_samples:
        # Signal is shorter than or equal to desired window — return full signal
        return 0, T

    cumsum = np.cumsum(rms_smooth)
    cumsum = np.insert(cumsum, 0, 0)   # prepend 0 for clean indexing

    # window_energy[i] = sum of rms_smooth[i : i + window_samples]
    window_energy = cumsum[window_samples:] - cumsum[:-window_samples]

    best_start = int(np.argmax(window_energy))
    best_end   = best_start + window_samples

    return best_start, best_end


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


def convert_dataset(dataset_dir, output_fpath, segment_gestures):
    # Reallocate subject ids. Originally, ids are repeated across training and testing splits
    # We want unique ids for each user to avoid conflicts when storing in the same HDF5 file.
    new_subject_id = 0     
    # EPN612 splits: 'trainingJSON' (train users) and 'testingJSON' (test users)
    with h5py.File(output_fpath, "w") as f:
        root = f.create_group("epn612")

        for mode in ["training", "testing"]:

            data_dir = dataset_dir / f"{mode}JSON"
            if not data_dir.exists(): continue

            user_dirs = list(data_dir.iterdir())
            for user_dir in tqdm(user_dirs, desc=f"Processing {mode} users"):
                if not user_dir.is_dir(): continue
                
                json_file = user_dir / f"{user_dir.name}.json"
                if not json_file.exists(): continue

                with open(json_file, "r") as j:
                    user_data = json.load(j)
                new_subject_id += 1

                # Metadata setup
                gen_info = user_data['generalInfo']
                user_info = user_data['userInfo']

                participant_id = f"sub-{new_subject_id:03d}"
                subj_group = root.create_group(participant_id)

                # Process metadata to get standardised versions
                user_std, session_std, run_std = process_metadata(user_info, gen_info, participant_id, json_file)

                # Resplit the subjects for a more standard train/val/test split. 
                if new_subject_id <= 428:
                    subj_split = "train"
                elif new_subject_id <= 489:
                    subj_split = "val"
                else:
                    subj_split = "test"

                user_std['inter_subject_split'] = subj_split

                # Store user metadata as attributes
                for k, v in user_std.items():
                    subj_group.attrs[k] = v

                # Create the session group
                # EPN does not explicitly state that each sample (idx_1, idx_2, etc.) is from a different recording session, 
                # so we treat them all being from the same session
                session_key = f'ses-001'  # Single session per user
                session_group = subj_group.create_group(session_key)
                for k, v in session_std.items():
                    session_group.attrs[k] = v

                # Again, reallocating session ids to avoid conflicts between training and testing splits
                new_run_id = 0

                # Gesture counter tracks how many times we've seen each gesture for the current user.
                # We reset between training and testing to get the same reps from both original splits in out new splits
                gesture_counter = {} 

                training_samples = user_data.get('trainingSamples', {})
                for original_run_id, run_data in training_samples.items():
                    new_run_id += 1
                    run_key = f"run-{new_run_id:03d}"
                    run_group = session_group.create_group(run_key)
                    origin_path = f'{str(json_file)}:trainingSamples.{original_run_id}'
                    process_run(run_std, run_group, run_data, origin_path, run_key, gesture_counter, segment_gestures)

                testing_samples = user_data.get('testingSamples', {})
                gesture_counter = {}  # Reset gesture counter for testing samples
                for original_run_id, run_data in testing_samples.items():
                    new_run_id += 1
                    run_key = f"run-{new_run_id:03d}"
                    run_group = session_group.create_group(run_key)
                    origin_path = f'{str(json_file)}:testingSamples.{original_run_id}'
                    process_run(run_std, run_group, run_data, origin_path, run_key, gesture_counter, segment_gestures)



def main(dataset_dir=None, output_dir=None, download_data=False, include_rest=False, segment_gestures=True, dataset_name="epn612"):
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
    parser.add_argument("--dataset_dir", type=str, default="data/epn612/")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the converted HDF5 files. Defaults to the dataset directory.")
    parser.add_argument("--download_data", action="store_true", help="Whether to download the EPN612 dataset.")
    parser.add_argument("--segment_gestures", action="store_true", help="Whether to segment the EMG data to only include the gesture execution period.")
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.download_data, args.segment_gestures)




"""
The dataset is organized as follows:

- data/epn612/
    - trainingJSON/
        - user_1/
            - photo.png
            - user1.json
        - user_2/
            ...
    - testingJSON/
        - user_1/
            - photo.png
            - user1.json
        - user_2/
            ...


Each user JSON file contains the following structure:
{
    "generalInfo": {
        "deviceModel": "Myo Armband",
        "samplingFrequencyInHertz": 200,
        "recordingTimeInSeconds": 5,
        "repetitionsForSynchronizationGesture": 5,
        "myoPredictionLabel": {
            "noGesture": 0,
            "fist": 1,
            "waveIn": 2,
            "waveOut": 3,
            "open": 4,
            "pinch": 5
        }
    },
    "userInfo": {
        "userName": <>,
        "age": <>,
        "gender": <>,
        "occupation": <>,
        "ethnicGroup": <>,
        "handedness": <>,
        "ArmDamage": <>,
        "distanceFromElbowToMyoInCm": <>,
        "distanceFromElbowToUlnaInCm": <>,
        "armPerimeterInCm": <>,
        "date": <>
    },
    "trainingSamples": {
        "idx_1": {
            "gestureName": <>,
            "startPointforGestureExecution": <>,
            "myoDetection": <>,
            "emg": <>,
            "accelerometer": <>,
            "gyroscope": <>
            quaternion": <>
        },
        ...
    "testingSamples": {
        (as above)

"""