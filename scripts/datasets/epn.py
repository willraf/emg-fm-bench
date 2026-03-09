"""Script epn.py

NOTE:
    There is a bit of a hack going on. In order to avoid repeating group names (e.g. session 1 from both training and testing samples)
    I add the split name to the session group name (e.g. train_session_1, test_session_1). This way we can store both training and testing samples for each user in the same HDF5 file without conflicts.
    This can be confusing as it refers to the SOURCE of the session, not the reallocated split. 
    The same goes for users naming.

Convert epn612 dataset from JSON to HDF5 format. The dataset is organized as follows:

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


        
Converts to HDF5 format with file structure:
- data/epn612/
    - training/
        training.hdf5
        testing.hdf5
    - testing/
        training.hdf5
        testing.hdf5

Each HDF5 file contains the following structure
- epn612 (root group)
    - user1
        - .attrs: name, age, gender, occupation, ethnicGroup, handedness, ArmDamage, distanceFromElboeToMyoInCm, 
                    distanceFromElbowToUlnaInCm, armPerimeterInCm, date, deviceModel, samplingFrequencyInHertz (of EMG), recordingTimeInSeconds,
                    repetitionsForSynchronizationGesture
        - session1
            - .attrs: gesture_name, start_point, session
            - emg (dataset): shape (T, 8), dtype float32
                - .attrs: sample_rate
            - imu (dataset): shape (T, 10) with compound dtype for accel, gyro, quat
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


def process_samples(sample_dict, user_group, session_info, original_session_split="train", subj_split="train"):
    """Processes a dictionary of samples and adds them to the user_group."""
    total_samples = len(sample_dict)
    val_split_idx = int(0.7 * total_samples)  # 70% for training, 10% for validation, 20% for testing
    test_split_idx = int(0.8 * total_samples)

    for sess_idx, (session_id, session_data) in enumerate(sample_dict.items()):
        
        # Update session ID to avoid naming conflicts between training and testing samples (e.g. session 1 in both)
        session_num = int(session_id.split('_')[-1])  # Extract numeric part of session ID
        # We offset the session number by 150 for test sessions to avoid conflicts with training sessions (e.g. session1 in training and testing)
        if original_session_split == "test":
            session_num += 150
        session_id = f'session{session_num}'
            
        # 1. Basic Metadata
        session_group = user_group.create_group(session_id)
        session_group.attrs['gesture_name'] = session_data.get('gestureName', 'unknown')
        session_group.attrs['gesture_id'] = get_global_gesture_id(get_gesture_id(session_group.attrs['gesture_name']))
        session_group.attrs['start_point'] = session_data.get('startPointforGestureExecution', 0)
        session_group.attrs['session_id'] = session_id

        for k, v in session_info.items():
            session_group.attrs[k] = v

        # Figure out session split
        # Unlabelled sessions (gesture_id == -1) go into a separate "unlabelled" split, otherwise we use the val/test/train split based on index
        if session_group.attrs['gesture_id'] == -1:
            sess_split = "unlabelled"
            subj_split = "unlabelled"
        elif sess_idx < val_split_idx:
            sess_split = "train"
        elif val_split_idx <= sess_idx < test_split_idx:
            sess_split = "val"
        else:
            sess_split = "test"

        session_group.attrs['intra_session_split'] = sess_split
        # EPN612 does not explictly state that sessions (idx_1, idx_2, etc.) are from different recording sessions, so we treat as intra-session only
        session_group.attrs['inter_session_split'] = "n/a"
        session_group.attrs['subject_split'] = subj_split

        # 2. EMG Dataset (T, 8)
        emg_dict = session_data.get('emg', {})
        # Shape (8, T)
        emg_data = np.array([emg_dict.get(f'ch{i}', []) for i in range(1, 9)], dtype=np.float32)
        
        # Dead sample check (optional but recommended)
        if emg_data.size == 0 or np.std(emg_data) == 0:
            continue

        emg_data /= 128.0  # Normalize to [-1, 1]
        emg_data = emg_data.T  # Shape (T, 8)
        session_group.create_dataset("emg", data=emg_data, dtype='<f4', compression="gzip")
        session_group["emg"].attrs['sample_rate'] = session_info.get('sample_rate', 200)

        accel = session_data.get('accelerometer', {})
        gyro = session_data.get('gyroscope', {})
        quat = session_data.get('quaternion', {})

        session_group.create_dataset("accelerometer", data=np.array([accel.get(axis, []) for axis in ['x', 'y', 'z']], dtype=np.float32).T, compression="gzip")
        session_group.create_dataset("gyroscope", data=np.array([gyro.get(axis, []) for axis in ['x', 'y', 'z']], dtype=np.float32).T, compression="gzip")
        session_group.create_dataset("quaternion", data=np.array([quat.get(axis, []) for axis in ['w', 'x', 'y', 'z']], dtype=np.float32).T, compression="gzip")

        session_group["accelerometer"].attrs['sample_rate'] = 50
        session_group["gyroscope"].attrs['sample_rate'] = 50
        session_group["quaternion"].attrs['sample_rate'] = 50

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


def process_metadata(user_info, general_info):
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
        'subject_id': str(user_info.get('name', 'unknown')),
        'age': int(user_info.get('age', -1)),
        'gender': gender,
        'handedness': handedness,
        'occupation': user_info.get('occupation', 'unknown'),
        'ethnicity': user_info.get('ethnicGroup', 'unknown'),
        'arm_damage': user_info.get('ArmDamage', 'unknown'),
        'distance_elbow_to_myo_cm': float(user_info.get('distanceFromElbowToMyoInCm', -1)),
        'distance_elbow_to_ulna_cm': float(user_info.get('distanceFromElbowToUlnaInCm', -1)),
        'forearm_circumference_cm': float(user_info.get('armPerimeterInCm', -1)),
    }

    standadised_session_info = {
        'device_model': general_info.get('deviceModel', 'unknown'),
        'sample_rate': int(general_info.get('samplingFrequencyInHertz', -1)),
        'recording_time_s': int(general_info.get('recordingTimeInSeconds', -1)),
        'repetitions_for_sync_gesture': int(general_info.get('repetitionsForSynchronizationGesture', -1)),
        'date': user_info.get('date', 'unknown')
    }

    return standardised_user_info, standadised_session_info


def download_epn612(data_dir):
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


def main(dataset_dir, output_dir=None, download_data=False):
    dataset_dir = Path(dataset_dir)

    if output_dir is None:
        output_fpath = dataset_dir / "epn612.hdf5"
    else:
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_fpath = output_dir / "epn612.hdf5"

    if download_data:
        download_epn612(dataset_dir)
    
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

                # Metadata setup
                gen_info = user_data['generalInfo']
                u_info = user_data['userInfo']
                # Get standardized metadata for user and session
                user_std, session_std = process_metadata(u_info, gen_info)

                subject_id = user_std['subject_id']
                subject_num = int(subject_id.replace("user", ""))  # Extract numeric part for splitting logic

                # Figure out the split this user should be in 
                if mode == "training":
                    # All users in trainingJSON are in the training split
                    subj_split = "train"
                else:
                    # users 1-122 go into training. users 123-183 go into val, users 184-306 go into test
                    if subject_num <= 122:
                        subj_split = "train"
                    elif 123 <= subject_num <= 183:
                        subj_split = "val"
                    else:
                        subj_split = "test"
                    # Offset the subject ID by the split to avoid conflicts (e.g. user_1 in training and testing)
                    subject_num += 306
                
                unique_subj_id = f'user{subject_num}'
                subj_group = root.create_group(unique_subj_id)

                # Store metadata as attributes
                for k, v in user_std.items():
                    subj_group.attrs[k] = v
                subj_group.attrs['subject_split'] = subj_split

                # Process the internal splits
                # user_data['trainingSamples'] goes to training.hdf5
                process_samples(user_data.get('trainingSamples', {}), subj_group, session_std, original_session_split="train", subj_split=subj_split)
                # user_data['testingSamples'] goes to testing.hdf5
                process_samples(user_data.get('testingSamples', {}), subj_group, session_std, original_session_split="test", subj_split=subj_split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/epn612/")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the converted HDF5 files. Defaults to the dataset directory.")
    parser.add_argument("--download_data", action="store_true", help="Whether to download the EPN612 dataset.")
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.download_data)