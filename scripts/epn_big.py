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
                    repetitionsForSynchronizationGesture, gesture_name, start_point, session
        - session1
            - .attrs: gesture_name, start_point, session
            - emg (dataset): shape (T, 8), dtype float32
            - imu (dataset): shape (T, 10) with compound dtype for accel, gyro, quat
"""

import h5py
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def process_samples(sample_dict, user_group, general_info):
    """Processes a dictionary of samples and adds them to the user_group."""
    for session_idx, session_data in sample_dict.items():
        # 1. Basic Metadata
        session_group = user_group.create_group(f'session_{session_idx}')
        session_group.attrs['gesture_name'] = session_data.get('gestureName', 'unknown')
        session_group.attrs['start_point'] = session_data.get('startPointforGestureExecution', 0)
        session_group.attrs['session'] = session_idx

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

        # 3. IMU Dataset (Structured)
        imu_dtype = np.dtype([
            ('accelerometer', '<f4', (3,)),
            ('gyroscope', '<f4', (3,)),
            ('quaternion', '<f4', (4,))
        ])

        accel = session_data.get('accelerometer', {})
        gyro = session_data.get('gyroscope', {})
        quat = session_data.get('quaternion', {})

        T_imu = min(len(accel.get('x', [])), len(gyro.get('x', [])), len(quat.get('w', [])))
        if T_imu == 0: continue

        imu_array = np.empty(T_imu, dtype=imu_dtype)
        for i, axis in enumerate(['x', 'y', 'z']):
            imu_array['accelerometer'][:, i] = accel.get(axis, [])[:T_imu]
            imu_array['gyroscope'][:, i] = gyro.get(axis, [])[:T_imu]
        for i, axis in enumerate(['w', 'x', 'y', 'z']):
            imu_array['quaternion'][:, i] = quat.get(axis, [])[:T_imu]

        session_group.create_dataset("imu", data=imu_array, compression="gzip")

def main(dataset_dir):
    dataset_dir = Path(dataset_dir)
    
    # EPN612 splits: 'trainingJSON' (train users) and 'testingJSON' (test users)
    for mode in ["training", "testing"]:
        data_dir = dataset_dir / f"{mode}JSON"
        if not data_dir.exists(): continue
        
        save_dir = dataset_dir / mode 
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

                # Create user groups in both split-files
                u_train = root_train.create_group(u_info['name'])
                u_test = root_test.create_group(u_info['name'])

                # Store attributes
                for group in [u_train, u_test]:
                    for k, v in u_info.items(): group.attrs[k] = str(v)
                    for k, v in gen_info.items():
                        if not isinstance(v, dict): group.attrs[k] = str(v)

                # Process the internal splits
                # user_data['trainingSamples'] goes to training.hdf5
                process_samples(user_data.get('trainingSamples', {}), u_train, gen_info)
                # user_data['testingSamples'] goes to testing.hdf5
                process_samples(user_data.get('testingSamples', {}), u_test, gen_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/epn612/")
    args = parser.parse_args()
    main(args.dataset_dir)