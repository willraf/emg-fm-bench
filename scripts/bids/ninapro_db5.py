""" Module ninapro_db5.py

Script to convert the Ninapro DB5 dataset into BIDS format

Usage:
    $ python -m scripts.bids.ninapro_db5 --dataset_dir data/ninapro_db5/ --output_dir data/ninapro_db5/ --download_data


Authors:
    Will Raftery

Date:
    29/04/2026
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
import json
import pyedflib

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
        1: "Exercise B",
        2: "Exercise C",
        3: "Exercise A",
    }
    return exercise_mapping.get(exercise_num, f"exercise_{exercise_num:02d}")


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
    with pyedflib.EdfWriter(str(emg_fpath), n_channels=emg.shape[1], file_type=pyedflib.FILETYPE_EDFPLUS) as edf_writer:
        channel_info = []
        for i in range(emg.shape[1]):
            channel_info.append({
                'label': f'EMG{i+1}',
                'dimension': 'arbitrary',
                'sample_rate': 200,  
                'physical_min': np.min(emg[:, i]),
                'physical_max': np.max(emg[:, i]),
                'digital_min': -128,
                'digital_max': 128,
            })
        edf_writer.setSignalHeaders(channel_info)
        edf_writer.writeSamples(emg)

        
def create_emg_json(emg_json_fpath, sampling_rate, duration, task_label):
    emg_json = {
        "EMGPlacementScheme": "Other",
        "EMGPlacementDescription": "Two, 8-channel Myo armbands placed laterally adjecent on the forearm. The upper Myo armband is placed closer to the elbow with the first electrode on the radio humeral joint. The lower Myo armband is placed just below the first, closer to the hand, tilted of 22.5°",
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
    }
    with open(emg_json_fpath, 'w') as f:
        json.dump(emg_json, f, indent=4)


def create_events(mat, output_dir):
    # Extract the relevant variables
    repetition = mat['repetition'].flatten()
    stimulus = mat['stimulus'].flatten()
    rerepetition = mat['rerepetition'].flatten()
    restimulus = mat['restimulus'].flatten()

    diff = np.diff(stimulus, prepend=-1)
    starts = np.where(diff != 0)[0]
    ends = np.append(starts[1:], len(stimulus))

    # Create events.tsv
    events_fpath = Path(output_dir) / "events.tsv"
    with open(events_fpath, 'w') as f:
        f.write("onset\tduration\ttrial_type\n")
        for i in range(len(repetition)):
            f.write(f"{repetition[i]}\t{stimulus[i]}\t{rerepetition[i]}_{restimulus[i]}\n")

def main(dataset_dir, output_dir, download_data):
    output_dir = Path(output_dir)
    if download_data:
        download_ninapro_db5(dataset_dir)
    
    print(f"Converting .mat files from {dataset_dir} to BIDS format in {output_dir}...")

    create_dataset_description(output_dir)
    create_readme(output_dir)
    create_participants_json(output_dir)
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
        create_events(mat, output_dir)

        # Save EMG data in edf
        emg = mat['emg'].T       # shape (n_channels, n_samples), this shape is expected by edf format
        emg_fpath = emg_dir / f"{base_filename}_emg.edf"
        write_emg(emg, emg_fpath)

        # Create the emg json sidecare
        sampling_rate = int(mat['frequency'][0][0])
        duration = emg.shape[1] / sampling_rate
        create_emg_json(emg_dir / f"{base_filename}_emg.json", sampling_rate, duration, task_label)

        acc = mat['acc'].astype(np.float32)
        glove = mat['glove'].astype(np.float32)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/ninapro_db5")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--download_data", action='store_true', help="If set, will attempt to download the raw .mat files from the official Ninapro DB5 source")
    args = parser.parse_args()
    
    main(args.dataset_dir, args.output_dir, args.download_data)