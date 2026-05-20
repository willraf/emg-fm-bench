""" Module ninapro_db5.py

Script to convert the Ninapro DB5 dataset into BIDS format

Usage:
    $ python -m scripts.bids.ninapro_db5 --dataset_dir data/ninapro_db5/ --output_dir data/ninapro_db5_bid/ --download_data


Authors:
    Will Raftery

Date:
    29/04/2026
"""

import scipy.io
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import urllib.request
import zipfile
import io
import json
import csv

import pyedflib
import pandas as pd

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


def create_dataset_description(output_dir):
    authors = ['Pizzolato, Stefano', 'Tagliapietra, Luca', 'Cognolato, Matteo', 'Reggiani, Monica', 'Müller, Henning', 'Atzori, Manfredo', 'Raftery, William']

    description = {
        "Name": "NinaPro DB5",
        "BIDSVersion": "1.11.1",
        "DatasetType": "raw",
        "License": "CC BY 4.0",
        "Authors": authors,
        "Keywords": ["EMG", "sEMG", "Kinematics", "Joint Angles", "Hand Movements", "Movement Classification", "Gesture Recognition", "Gesture Classification", "Myoelectric Signals"],
        "HowToAcknowledge": "Please cite the original NinaPro DB5 paper when using this dataset.",
        "ReferencesAndLinks": [
            "https://ninapro.hevs.ch/instructions/DB5.html", "Pizzolato et al., "
            "Comparison of six electromyography acquisition setups on hand movement classification tasks, PLOS One, 2017"],
        "DatasetDOI": "https://doi.org/10.5281/zenodo.1000116",
        "GeneratedBy": [
            {
                "Name": "NinaPro to BIDS Conversion Script",
                "Version": "1.0.0",
                "Description": "Custom Python script to convert NinaPro DB5 from its native format to BIDS.",
                "CodeURL": "https://github.com/willraf/emg-stream/tree/main/scripts/bids/ninapro_db5.py"
            }
        ],
        "SourceDatasets": [
            {
                "URL": "https://ninapro.hevs.ch/instructions/DB5.html",
                "DOI": "doi:10.5281/zenodo.1000116"
            }
        ]
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'dataset_description.json', 'w') as f:
        json.dump(description, f, indent=4)


def create_readme(output_dir):
    readme_content = """
    # NinaPro DB5 Dataset in BIDS Format

    This dataset contains the Ninapro DB5 hand movement data converted into BIDS format. The original dataset can be found at [NinaPro DB5](https://ninapro.hevs.ch/instructions/DB5.html).

    ## Dataset Structure

    The dataset is organized according to the BIDS specification, with the following structure:

    ```
    /dataset_description.json
    /README
    /sub-01/
        emg/
        motion/
    /sub-02/
        emg/
        motion/
    ...
    ```

    Each subject folder contains the EMG and kinematics data for that subject, with filenames following the BIDS convention.
    No session folders are used as the data was collected in a single session per subject.

    ## Acknowledgements

    Please cite the original NinaPro DB5 paper when using this dataset:

    Pizzolato et al., "Comparison of six electromyography acquisition setups on hand movement classification tasks," PLOS One, 2017.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'README', 'w') as f:
        f.write(readme_content)


def create_participants_json(output_dir):
    participants = {
        "age":
        {
            "Description": "Age of the participant at the time of data collection",
            "Units": "years"
        },
        "sex":
        {
            "Description": "Sex of the participant",
            "Levels": {
                "m": "Male",
                "f": "Female",
                "o": "Other",
                "u": "Unknown"
            }
        },
        "handedness":
        {
            "Description": "Handedness of the participant",
            "Levels": {
                "r": "Right-handed",
                "l": "Left-handed",
                "a": "Ambidextrous",
                "u": "Unknown"
            }
        },
        "weight":
        {
            "Description": "Weight of the participant at the time of data collection",
            "Units": "kg"
        },
        "height":
        {
            "Description": "Height of the participant at the time of data collection",
            "Units": "cm"
        },
        "forearm_circumference":
        {
            "Description": "Circumference of the participant's forearm at the time of data collection",
            "Units": "cm"
        }
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'participants.json', 'w') as f:
        json.dump(participants, f, indent=4)


def create_events_json(output_dir, exercise_name):
    events = {
        "onset": {
            "Description": "Time of the event onset",
            "Units": "s"
        },
        "duration": {
            "Description": "Duration of the event",
            "Units": "s"
        },
        "sample": {
            "Description": "Event onset sample (at 200Hz)",
            "Units": "samples"
        },
        "sample_duration": {
            "Description": "Event duration in samples (at 200Hz)",
            "Units": "samples"
        },
        "trial_type": {
            "LongName": "Event Category",
            "Description": "The type of event that occured.",
            "Levels": {
                "prompt": "stimulus/repetition aligned event (the cue shown to the subject)",
                "action": "restimulus/rerepetition aligned event (the actual movement onset)",
            }
        },
        "repetition": {
            "Description": "The repetition number of the movement (1-6)",
            "Units": "repetitions"
        },
        "gesture_id": {
            "Description": "The global gesture number (0-52, where 1-52 are movements and 0 is rest). Gestures 1-12 are from Exercise A, 13-29 are from Exercise B, and 30-52 are from Exercise C.",
            "Levels": {
                0: "rest",
                1: "Index flexion",
                2: "Index extension",
                3: "Middle flexion",
                4: "Middle extension",
                5: "Ring flexion",
                6: "Ring extension",
                7: "Little flexion",
                8: "Little extension",
                9: "Thumb adduction",
                10: "Thumb abduction",
                11: "Thumb flexion",
                12: "Thumb extension",
                13: "Thumb up",
                14: "Extension of index and middle, flexion of others",
                15: "Flexion of ring and little finger, extension of others",
                16: "Thumb opposing base of little finger",
                17: "Abduction of all fingers",
                18: "Fingers flexed together in fist",
                19: "Pointing index",
                20: "Adduction of extended fingers",
                21: "Wrist supination (axis: middle finger)",
                22: "Wrist pronation (axis: middle finger)",
                23: "Wrist supination (axis: little finger)",
                24: "Wrist pronation (axis: little finger)",
                25: "Wrist flexion",
                26: "Wrist extension",
                27: "Wrist radial deviation",
                28: "Wrist ulnar deviation",
                29: "Wrist extension with closed hand",
                30: "Large diameter grasp",
                31: "Small diameter grasp (power grip)",
                32: "Fixed hook grasp",
                33: "Index finger extension grasp",
                34: "Medium wrap",
                35: "Ring grasp",
                36: "Prismatic four fingers grasp",
                37: "Stick grasp",
                38: "Writing tripod grasp",
                39: "Power sphere grasp",
                40: "Three finger sphere grasp",
                41: "Precision sphere grasp",
                42: "Tripod grasp",
                43: "Prismatic pinch grasp",
                44: "Tip pinch grasp",
                45: "Quadpod grasp",
                46: "Lateral grasp",
                47: "Parallel extension grasp",
                48: "Extension type grasp",
                49: "Power disk grasp",
                50: "Open a bottle with a tripod grasp",
                51: "Turn a screw (grasp the screwdriver with a stick grasp)",
                52: "Cut something (grasp the knife with an index finger extension grasp)"
            }
        }
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'task-{exercise_name}_events.json'

    with open(output_path, 'w') as f:
        json.dump(events, f, indent=4)


def create_participants_tsv(participants, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'participants.tsv', 'w') as f:
        # Write header
        f.write("participant_id\tage\tsex\thandedness\tweight\theight\tforearm_circumference\n")
        for participant_id, metadata in participants.items():
            f.write(f"{participant_id}\t{metadata['age']}\t{metadata['sex']}\t{metadata['handedness']}\t{metadata['weight']}\t{metadata['height']}\t{metadata['forearm_circumference']}\n")


def exercise_num_to_name(exercise_num):
    """Mapping are pretty weird, but we're mapping to names given in docs"""
    exercise_mapping = {
        1: "ExerciseB",
        2: "ExerciseC",
        3: "ExerciseA",
    }
    return exercise_mapping.get(exercise_num, f"exercise_{exercise_num:02d}")


def task_name_to_description(task_name):
    task_mapping = {
        "ExerciseA": "12 basic hand movements (flexions/extensions of fingers and thumb)",
        "ExerciseB": "8 hand/wrist movements and 9 wrist postures",
        "ExerciseC": "22 grasping tasks and 1 open/close hand task",
    }
    return task_mapping.get(task_name, "n/a")


def get_participant_metadata(mat):
    """Extract participant metadata from the .mat file. The .mat files don't have much metadata, so we'll just return n/a for everything."""

    # Standardize gender
    gender = str(mat['gender'][0]).lower()
    if 'm' in gender: gender = 'm'
    elif 'f' in gender: gender = 'f'
    else: gender = 'u'

    # Standardize laterality -> handedness
    lat = str(mat['laterality'][0]).lower()
    handedness = 'r' if 'r' in lat else ('l' if 'l' in lat else 'u')

    metadata = {
        'age': int(mat['age'][0][0]),
        'sex': gender,
        'handedness': handedness,
        'weight': float(mat['weight'][0][0]),
        'height': float(mat['height'][0][0]),
        'forearm_circumference': float(mat['circumference'][0][0]),
    }
    return metadata
        

def write_emg(emg, emg_fpath):
    """
    Args:
        emg: Shape (C, T) numpy array containing the EMG data
        emg_fpath: Path to save the EDF file to. Must be *_emg.edf
    """
    n_channels = emg.shape[0]
    with pyedflib.EdfWriter(str(emg_fpath), n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as edf_writer:
        channel_info = []
        for i in range(n_channels):
            channel_info.append({
                'label': f'EMG{i+1}',
                'dimension': 'arbitrary',
                'sample_frequency': 200,  
                'physical_min': -1.0,
                'physical_max':  1.0,
                'digital_min':  -32768,
                'digital_max':   32767,
            })
        edf_writer.setSignalHeaders(channel_info)
        edf_writer.writeSamples(emg)

        
def create_emg_json(emg_json_fpath, sampling_rate, duration, task_label):
    emg_json = {
        "EMGPlacementScheme": "Other",
        "EMGPlacementSchemeDescription": "Two, 8-channel Myo armbands placed laterally adjecent on the forearm. The upper Myo armband is placed closer to the elbow with the first electrode on the radio humeral joint. The lower Myo armband is placed just below the first, closer to the hand, tilted of 22.5°",
        "EMGReference": "Bipolar",
        "SamplingFrequency": sampling_rate,
        "PowerLineFrequency": 50,
        "RecordingType": "continuous",
        "SoftwareFilters": "n/a",
        "EMGChannelCount": 16,
        "HardwareFilters": {
            "Notch Filter": {
                "Frequency": 50,
            },
        },
        "RecordingDuration": duration,
        "ElectrodeMaterial": "stainless steel",
        "ElectrodeType": "surface",
        "Manufacturer": "Thalmic Labs",
        "ManufacturersModelName": "Myo Armband",
        "TaskName": task_label,
        "TaskDescription": task_name_to_description(task_label)
    }
    with open(emg_json_fpath, 'w') as f:
        json.dump(emg_json, f, indent=4)


def get_global_gesture_id(gesture_id, exercise_id):
    """Maps the gesture_id (which is local to each exercise) to a global gesture id (which is unique across all exercises)
    
    NinaPro DB5 has a unique gesture distribution across exercises:
    
    Each exercise has a different set of gestures:

    Exercise 1 (2 in fname, B in docs): 17 gestures (1-17)
    Exercise 2 (3 in fname, C in docs): 23 gestures (18-40)
    Exercise 3 (1 in fname, A in docs): 12 gestures (41-52)
    """
    if gesture_id == 0:
        return 0  # rest is always 0
    
    if exercise_id == 3:  # Exercise A
        return gesture_id  # 1-12 map to 1-12
    elif exercise_id == 1:  # Exercise B
        return gesture_id + 12  # 1-17 map to 13-29 
    elif exercise_id == 2:  # Exercise C
        return gesture_id + 12 + 17  # 1-23 map to 30-52
    else:
        raise ValueError(f"Unknown exercise_id {exercise_id} for gesture_id mapping")


def extract_events(label_array, rep_array, trial_type_label, frequency, exercise_id):
    """
    Helper funciton to extract events from the stimulus and repetition arrays. 
    Can be used for both prompt (stimulus/repetition) and action (restimulus/rerepetition) events
    """
    diff = np.diff(label_array, prepend=-1)
    starts = np.where(diff != 0)[0]
    ends = np.append(starts[1:], len(label_array))

    rows = []

    for start, end in zip(starts, ends):
        stim_val = int(label_array[start])
        rep_val  = int(rep_array[start])

        # Map stimulus value to gesture_id (0 for rest, 1-12 for Exercise A, 13-29 for Exercise B, 30-52 for Exercise C)
        global_gesture_id = get_global_gesture_id(stim_val, exercise_id)

        rows.append({
            "onset": round(start / frequency, 6),
            "duration": round((end - start) / frequency, 6),
            "sample": start,
            "sample_duration": end - start,
            "trial_type": trial_type_label,
            "gesture_id": global_gesture_id,
            "repetition": rep_val,
        })

    return rows


def create_events_tsv(mat, filepath):
    # Extract the relevant variables
    repetition   = mat['repetition'].flatten().astype(int)
    stimulus     = mat['stimulus'].flatten().astype(int)
    rerepetition = mat['rerepetition'].flatten().astype(int)
    restimulus   = mat['restimulus'].flatten().astype(int)
    frequency    = int(mat['frequency'][0][0])
    exercise_id  = int(mat['exercise'][0][0])

    rows = []

    rows.extend(extract_events(stimulus, repetition, "prompt", frequency, exercise_id))
    rows.extend(extract_events(restimulus, rerepetition, "action", frequency, exercise_id))
    # Sort by onset time
    rows = sorted(rows, key=lambda x: x['onset'])
    
    df = pd.DataFrame(rows, columns=["onset", "duration", "sample", "sample_duration", "trial_type", "gesture_id", "repetition"])
    
    # Save to all fpaths if its a list
    if isinstance(filepath, list):
        for fpath in filepath:
            df.to_csv(fpath, sep='\t', index=False)
    else:
        df.to_csv(filepath, sep='\t', index=False)


def create_motion_json(sampling_rate: int,
                        task_name: str, 
                        manufacturer: str, 
                        model_name: str, 
                        tracking_system: str,
                        channel_counts: dict, 
                        filepath: Path):

    metadata = {
        "SamplingFrequency": sampling_rate,
        "TaskName": task_name,
        "TaskDescription": task_name_to_description(task_name),
        "Manufacturer": manufacturer,
        "ManufacturersModelName": model_name,
        "TrackingSystemName": tracking_system
    }

    metadata.update(channel_counts)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

# def create_motion_channels_tsv(filepath: Path, 
#                                 channels: List[Dict[str, str]]):
#     """
#     Create BIDS *_channels.tsv file for motion data, which describes the individual channels in the motion data files.

#     Args:
#         filepath (Path): The path to the channels.tsv file to create.
#         channels (List[Dict[str, str]]): A list of dictionaries, each describing a channel with keys: 'name', 'component', 'type', 'tracked_point', 'units'.
#     """
#     df = pd.DataFrame(channels)
#     print(df)
#     df.to_csv(filepath, sep='\t', index=False)

def create_acc_channels_tsv(filepath: Path):
    headers = ["name", "component", "type", "tracked_point", "units"]
    channels = [
        ["acc_x", "x", "ACCEL", "forearm", "m/s^2"],
        ["acc_y", "y", "ACCEL", "forearm", "m/s^2"],
        ["acc_z", "z", "ACCEL", "forearm", "m/s^2"]
    ]

    df = pd.DataFrame(channels, columns=headers)
    df.to_csv(filepath, sep='\t', index=False)


def create_glove_channels_tsv(filepath, num_sensors=22):
    """Generates the _channels.tsv for the Cyberglove joint angles."""
    headers = ["name", "component", "type", "tracked_point", "units"]
    channels = []
    
    # Type MUST be JNTANG (Joint Angle) [8] and component is n/a for angles
    for i in range(num_sensors):
        channels.append([
            f"joint_{i+1}", 
            "n/a", 
            "JNTANG", 
            f"glove_sensor_{i+1}", 
            "deg"
        ])
        
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        writer.writerows(channels)

def main(dataset_dir, output_dir, download_data=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(dataset_dir)

    if download_data:
        download_ninapro_db5(dataset_dir)
    
    print(f"Converting .mat files from {dataset_dir} to BIDS format in {output_dir}...")

    create_dataset_description(output_dir)
    create_readme(output_dir)
    create_participants_json(output_dir)
    for exercise_name in ["ExerciseA", "ExerciseB", "ExerciseC"]:
        create_events_json(output_dir, exercise_name)
    # Create data structure to track participants
    participants = {}

    # Should work with both nested and flat dir structures
    mat_files = sorted(dataset_dir.rglob("*.mat"))
    for fpath in tqdm(mat_files, desc="Processing .mat files"):
        
        mat = scipy.io.loadmat(str(fpath))

        subject_num = mat['subject'][0][0]
        participant_id = f"sub-{subject_num:02d}"

        # Check if participant_id already exists in participants.tsv, if not add it with empty metadata (since we don't have that info)
        if participant_id not in participants:
            participants[participant_id] = get_participant_metadata(mat)

        exercise_num = mat['exercise'][0][0]
        task_label = exercise_num_to_name(exercise_num)

        base_filename = f"{participant_id}_task-{task_label}"

        emg_dir = Path(output_dir) / participant_id / "emg"
        motion_dir = Path(output_dir) / participant_id / "motion"
        emg_dir.mkdir(parents=True, exist_ok=True)
        motion_dir.mkdir(parents=True, exist_ok=True)

        # Create the events tsv and json sidecar from the repetition, stimulus, rerepetition, and restimulus variables
        events_fpath = [
            emg_dir / f"{base_filename}_events.tsv",
            motion_dir / f"{base_filename}_events.tsv"
        ]
        create_events_tsv(mat, events_fpath)

        # Save EMG data in edf
        emg = mat['emg'].astype(np.float32).T  / 128   # (T, C) -> (C, T) normalise to [-1, 1] range since Myo armband has 8-bit resolution and range of [-128, 127]

        emg_fpath = emg_dir / f"{base_filename}_emg.edf"
        write_emg(emg, emg_fpath)

        # Create the emg json sidecare
        sampling_rate = int(mat['frequency'][0][0])
        duration = emg.shape[1] / sampling_rate
        create_emg_json(emg_dir / f"{base_filename}_emg.json", sampling_rate, duration, task_label)

        # Save kinematics data in tsv
        acc = mat['acc'].astype(np.float32)
        glove = mat['glove'].astype(np.float32)
        
        # Save acc and glove data in separate files
        acc_base = f"{base_filename}_tracksys-myoband"
        create_motion_json(sampling_rate, 
                           task_label,
                           "Thalmic Labs", 
                           "Myo Armband", 
                           "Accelerometer from sEMG band on right forearm", 
                           {"ACCELChannelCount": acc.shape[1]}, 
                           motion_dir / f"{acc_base}_motion.json")
        
        create_acc_channels_tsv(motion_dir / f"{acc_base}_channels.tsv")
        # Actually save the data
        np.savetxt(motion_dir / f"{acc_base}_motion.tsv", acc, delimiter='\t')

        # save glove data
        glove_base = f"{base_filename}_tracksys-glove"
        create_motion_json(sampling_rate,
                           task_label,
                           "CyberGlove Systems LLC", 
                           "CyberGlove II", 
                           "Data glove with 22 sensors on the right hand", 
                           {"JNTANGChannelCount": glove.shape[1]}, 
                           motion_dir / f"{glove_base}_motion.json")
        create_glove_channels_tsv(motion_dir / f"{glove_base}_channels.tsv")
        np.savetxt(motion_dir / f"{glove_base}_motion.tsv", glove, delimiter='\t')

    create_participants_tsv(participants, output_dir)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/ninapro_db5")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--download_data", action='store_true', help="If set, will attempt to download the raw .mat files from the official Ninapro DB5 source")
    args = parser.parse_args()
    
    main(args.dataset_dir, args.output_dir, args.download_data)