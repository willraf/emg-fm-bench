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
    - trainingHDF5/
        training/
            - user_1_idx_1.hdf5
            - user_1_idx_2.hdf5
            - ...
        testing/
            - user_1_idx_1.hdf5
            - user_1_idx_2.hdf5
            - ...
    - testingHDF5/
        training/
            - user_1_idx_1.hdf5
            - user_1_idx_2.hdf5
            - ...
        testing/
            - user_1_idx_1.hdf5
            - user_1_idx_2.hdf5
            - ...

Each HDF5 file contains the following structure
- epn612 (root group)
    - .attrs: name, age, gender, occupation, ethnicGroup, handedness, ArmDamage, distanceFromElboeToMyoInCm, 
                distanceFromElbowToUlnaInCm, armPerimeterInCm, date, deviceModel, samplingFrequencyInHertz (of EMG), recordingTimeInSeconds,
                repetitionsForSynchronizationGesture, gesture_name, start_point, session
    - emg (dataset): shape (8, T), dtype float32
    - imu (group)
        - accelerometer (dataset): shape (3, T), dtype float32
        - gyroscope (dataset): shape (3, T), dtype float32
        - quaternion (dataset): shape (4, T), dtype float32
"""

from pathlib import Path
import json
import h5py
import numpy as np
import argparse

def create_hdf5_for_session(session_data, user_info, general_info, fpath):
    """Create an HDF5 file for a single session.
    
    Structure:
        - epn612 (root group)
            - .attrs: name, age, gender, occupation, ethnicGroup, handedness, ArmDamage, distanceFromElboeToMyoInCm, 
                        distanceFromElbowToUlnaInCm, armPerimeterInCm, date, deviceModel, samplingFrequencyInHertz (of EMG), recordingTimeInSeconds,
                        repetitionsForSynchronizationGesture, gesture_name, start_point, session
            - emg (dataset): shape (8, T), dtype float32
            - imu (group)
                - accelerometer (dataset): shape (3, T), dtype float32
                - gyroscope (dataset): shape (3, T), dtype float32
                - quaternion (dataset): shape (4, T), dtype float32
    
    Args:
        session_data (dict): The data for the session.
            Keys: gestureName, startPointforGestureExecution, emg (dict of ch1-ch8), accelerometer (dict of x,y,z), gyroscope (dict of x,y,z), quaternion (dict of w,x,y,z)
        user_info (dict): The user information to store as attributes.
        general_info (dict): The general information to store as attributes.
        fpath (Path): The path to save the HDF5 file.
    """

    fpath.parent.mkdir(exist_ok=True, parents=True)

    f = h5py.File(fpath, 'w')

    root = f.create_group("epn612")

    # 1. Metadata (.attrs)
    for k, v in user_info.items():
        root.attrs[k] = v
    for k, v in general_info.items():
        # Handle nested dicts in generalInfo if necessary
        if not isinstance(v, dict):
            root.attrs[k] = v


    root.attrs['gesture_name'] = session_data.get('gestureName', 'unknown')
    root.attrs['start_point'] = session_data.get('startPointforGestureExecution', 0)
    root.attrs['session'] = session_data.get('session', 'unknown')

    # 2. EMG Dataset (8, ~1000)
    # Extracting from the nested ch1-ch8 structure
    emg_dict = session_data.get('emg', {})
    emg_data = np.array([emg_dict.get(f'ch{i}', []) for i in range(1, 9)], dtype=np.float32)
    # Divide by 128 to have values in range [-1, 1]
    emg_data /= 128.0
    emg_data = emg_data.T  # Transpose to shape (T, 8) for better compression and compatibility with PyTorch (time major)
    root.create_dataset("emg", data=emg_data, dtype='<f4', compression="gzip")      # Could use lzf

    # 1. Define the Structured Dtype (matching '<f4' precision from emg2pose)
    imu_dtype = np.dtype([
        ('accelerometer', '<f4', (3,)),  # [x, y, z]
        ('gyroscope', '<f4', (3,)),      # [x, y, z]
        ('quaternion', '<f4', (4,))      # [w, x, y, z]
    ])

    # 2. Extract raw lists from JSON data
    accel = session_data.get('accelerometer', {})
    gyro = session_data.get('gyroscope', {})
    quat = session_data.get('quaternion', {})

    # Determine the number of IMU samples (T)
    T_accel = len(accel.get('x', []))
    T_gyro = len(gyro.get('x', []))
    T_quat = len(quat.get('w', []))

    T_imu = min(T_accel, T_gyro, T_quat)

    # crop to the minimum length to ensure alignment
    accel = {axis: accel.get(axis, [])[:T_imu] for axis in ['x', 'y', 'z']}
    gyro = {axis: gyro.get(axis, [])[:T_imu] for axis in ['x', 'y', 'z']}
    quat = {axis: quat.get(axis, [])[:T_imu] for axis in ['w', 'x', 'y', 'z']}

    # Calculate sampling frequency of IMU and add to attributes
    recording_time = general_info.get('recordingTimeInSeconds', 1)
    if T_imu > 1 and recording_time > 0:
        imu_sampling_freq = T_imu / recording_time
        root.attrs['imuSamplingFrequencyInHertz'] = imu_sampling_freq

    # 3. Create an empty structured numpy array
    imu_array = np.empty(T_imu, dtype=imu_dtype)

    # 4. Fill the array
    # Accelerometer
    imu_array['accelerometer'][:, 0] = accel.get('x', [])
    imu_array['accelerometer'][:, 1] = accel.get('y', [])
    imu_array['accelerometer'][:, 2] = accel.get('z', [])

    # Gyroscope
    imu_array['gyroscope'][:, 0] = gyro.get('x', [])
    imu_array['gyroscope'][:, 1] = gyro.get('y', [])
    imu_array['gyroscope'][:, 2] = gyro.get('z', [])

    # Quaternion
    imu_array['quaternion'][:, 0] = quat.get('w', [])
    imu_array['quaternion'][:, 1] = quat.get('x', [])
    imu_array['quaternion'][:, 2] = quat.get('y', [])
    imu_array['quaternion'][:, 3] = quat.get('z', [])

    # 5. Save as a single dataset
    root.create_dataset("imu", data=imu_array, compression="gzip")

    # # 3. IMU Datasets (accelerometer, gyroscope, quaternion). I believe sampled at 50Hz
    # imu = root.create_group("imu")

    # # Accelerometer (3, ~250) - [x, y, z], T
    # accel_dict = session_data.get('accelerometer', {})
    # accel_data = np.array([accel_dict.get(axis, []) for axis in ['x', 'y', 'z']], dtype=np.float32)
    # imu.create_dataset("accelerometer", data=accel_data, dtype='<f4', compression="gzip")

    # # Gyroscope (3, ~250) - [x, y, z], T
    # gyro_dict = session_data.get('gyroscope', {})
    # gyro_data = np.array([gyro_dict.get(axis, []) for axis in ['x', 'y', 'z']], dtype=np.float32)
    # imu.create_dataset("gyroscope", data=gyro_data, dtype='<f4', compression="gzip")

    # # Quaternion (4, ~250) - [w, x, y, z], T
    # quat_dict = session_data.get('quaternion', {})
    # quat_data = np.array([quat_dict.get(axis, []) for axis in ['w', 'x', 'y', 'z']], dtype=np.float32)
    # imu.create_dataset("quaternion", data=quat_data, dtype='<f4', compression="gzip")

    f.close()


def main(dataset_dir):

    for mode in ["training", "testing"]:

        data_dir = dataset_dir / f"{mode}JSON"
        save_dir = dataset_dir / f"{mode}HDF5"
        save_dir.mkdir(exist_ok=True)


        for user_dir in data_dir.iterdir():
            user_name = user_dir.name
            
            json_file = user_dir / f"{user_name}.json"

            if not json_file.exists():
                continue

            with open(json_file, "r") as f:
                user_data = json.load(f)

            general_info = user_data['generalInfo']
            user_info = user_data['userInfo']

            training_samples = user_data['trainingSamples']

            for session_idx, session_data in training_samples.items():
                
                # Create HDF5 file for this session
                fname = f'{user_name}_{session_idx}.hdf5'
                fpath = save_dir / "training" / fname
                session_data['session'] = session_idx

                create_hdf5_for_session(session_data, user_info, general_info, fpath)
            
            testing_samples = user_data['testingSamples']

            for session_idx, session_data in testing_samples.items():
                
                # Create HDF5 file for this session
                fname = f'{user_name}_{session_idx}.hdf5'
                fpath = save_dir / "testing" / fname
                session_data['session'] = session_idx

                create_hdf5_for_session(session_data, user_info, general_info, fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EPN612 dataset from JSON to HDF5 format")
    parser.add_argument("--dataset_dir", type=str, default="data/epn612/")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    main(dataset_dir)