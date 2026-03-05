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

TRAIN_SUBJS = [1, 3, 4, 6, 8, 9, 10]
VAL_SUBJS = [11]
TEST_SUBJS = [5, 7]

INCLUDE_REST = True

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

def convert_ninapro_db5(dataset_dir, output_file):
    dataset_dir = Path(dataset_dir)
    mat_files = sorted(dataset_dir.glob("*.mat"))

    with h5py.File(output_file, 'w') as f:
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
                    subj_group.attrs['split'] = 'train'
                elif subject_id_int in VAL_SUBJS:
                    subj_group.attrs['split'] = 'val'
                elif subject_id_int in TEST_SUBJS:
                    subj_group.attrs['split'] = 'test'
                else:
                    subj_group.attrs['split'] = 'unknown'
                    
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
                    sample_group.attrs['split'] = 'val'
                elif rep_idx == 5:  # Test repetition
                    sample_group.attrs['split'] = 'test'
                else:
                    sample_group.attrs['split'] = 'train'
                
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

            
            # # Group by refined trial (rerepetition)
            # unique_reps = np.unique(repetition)
            # unique_stimuli = np.unique(stimulus)
            
            # # Ignore rest periods
            # valid_mask = (repetition > 0) & (stimulus > 0)
            # pairs = np.unique(np.vstack((repetition[valid_mask], stimulus[valid_mask])).T, axis=0)

            # for rep_idx, stim_idx in pairs:
                
            #     mask = ((repetition == rep_idx) & (stimulus == stim_idx))
            #     if not np.any(mask):
            #         continue
                
            #     exercise_id = sess_std['exercise_id']
            #     gesture_id = get_gesture_id(exercise_id, stim_idx)

            #     # Create sample group (E=Exercise, R=Repetition, G=Gesture)
            #     sample_name = f"E{exercise_id}_R{rep_idx}_G{gesture_id}"
            #     sample_group = subj_group.create_group(sample_name)
                
            #     # Per-sample Metadata
            #     sample_group.attrs['gesture_id'] = gesture_id
            #     sample_group.attrs['repetition_id'] = rep_idx

            #     if rep_idx == 2:  # Validation repetition
            #         sample_group.attrs['split'] = 'val'
            #     elif rep_idx == 5:  # Test repetition
            #         sample_group.attrs['split'] = 'test'
            #     else:
            #         sample_group.attrs['split'] = 'train'
                
            #     for k, v in sess_std.items():
            #         sample_group.attrs[k] = v

            #     # All modalities are oversampled to 200Hz, so they should be aligned by index
                
            #     # 1. EMG (T, 16) 
            #     sample_group.create_dataset("emg", data=emg[mask], compression="lzf")
            #     sample_group["emg"].attrs['sample_rate'] = sess_std.get('sample_rate', 200)
            #     sample_group["emg"].attrs['units'] = 'normalized'
            #     sensor_val = mat.get('sensor', ['unknown'])[0]
            #     # Use np.string_ to ensure it's HDF5-compatible (fixed-width)
            #     sample_group["emg"].attrs['sensor_type'] = np.string_(str(sensor_val))
                
            #     # 2. Accel (T, 3)
            #     sample_group.create_dataset("accel", data=acc[mask], compression="lzf")
            #     sample_group["accel"].attrs['sample_rate'] = sess_std.get('sample_rate', 200)
            #     sample_group["accel"].attrs['sensor_type'] = np.string_(str(sensor_val))
                
            #     # 3. Glove (T, 22) - Kinematic ground truth
            #     sample_group.create_dataset("glove", data=glove[mask], compression="lzf")
            #     sample_group["glove"].attrs['sample_rate'] = sess_std.get('sample_rate', 200)
            #     sample_group["glove"].attrs['sensor_type'] = 'cyber_glove'
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/ninapro_db5")
    parser.add_argument("--output_dir", type=str, default="data/ninapro_db5/ninapro_db5.hdf5")
    args = parser.parse_args()
    
    convert_ninapro_db5(args.dataset_dir, args.output_dir)