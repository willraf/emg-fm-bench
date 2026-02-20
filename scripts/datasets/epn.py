"""Script epn.py

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

def process_samples(sample_dict, user_group, session_info):
    """Processes a dictionary of samples and adds them to the user_group."""
    for session_idx, session_data in sample_dict.items():
        # 1. Basic Metadata
        session_group = user_group.create_group(f'session_{session_idx}')
        session_group.attrs['gesture_name'] = session_data.get('gestureName', 'unknown')
        session_group.attrs['gesture_id'] = get_global_gesture_id(get_gesture_id(session_group.attrs['gesture_name']))
        session_group.attrs['start_point'] = session_data.get('startPointforGestureExecution', 0)
        session_group.attrs['session_id'] = session_idx

        for k, v in session_info.items():
            session_group.attrs[k] = v

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


def main(dataset_dir, output_dir=None):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir) if output_dir else dataset_dir
    
    # EPN612 splits: 'trainingJSON' (train users) and 'testingJSON' (test users)
    for mode in ["training", "testing"]:
        data_dir = dataset_dir / f"{mode}JSON"
        if not data_dir.exists(): continue
        
        save_dir = output_dir / mode 
        save_dir.mkdir(exist_ok=True, parents=True)

        # Paths for the two sub-files per user-split
        train_h5_path = save_dir / "training.hdf5"
        test_h5_path = save_dir / "testing.hdf5"


        with h5py.File(train_h5_path, 'w') as f_train, h5py.File(test_h5_path, 'w') as f_test:
            # Create root groups
            root_train = f_train.create_group("epn612")
            root_test = f_test.create_group("epn612")

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
                

                # Create user groups in both split-files
                u_train = root_train.create_group(user_std['subject_id'])
                u_test = root_test.create_group(user_std['subject_id'])

                # Store attributes
                for group in [u_train, u_test]:
                    for k, v in user_std.items():
                        group.attrs[k] = v

                # Process the internal splits
                # user_data['trainingSamples'] goes to training.hdf5
                process_samples(user_data.get('trainingSamples', {}), u_train, session_std)
                # user_data['testingSamples'] goes to testing.hdf5
                process_samples(user_data.get('testingSamples', {}), u_test, session_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/epn612/")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the converted HDF5 files. Defaults to the dataset directory.")
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir)