""" Script neuromotor_wrist.py

Convert Neuromotor Wrist from its original HDF5 format (with one file per recording) into a structured HDF5 format for efficient loading in our EMG processing pipeline.

Usage:
    % python -m scripts.datasets.neuromotor_wrist --dataset_dir data/neuromotor_wrist/ --output_dir data/neuromotor_wrist/ --download_data

Authors:
    Will Raftery

Date:
    01/07/2026
"""


import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import requests
from tqdm import tqdm
import tarfile
from datetime import datetime
import pandas as pd

## -----------------------------------------------------------------
# Following is copyed from meta code 
# https://github.com/facebookresearch/generic-neuromotor-interface
# ------------------------------------------------------------------
def download_file(url, output_path, description=None):
    """
    Download a file from a URL with a progress bar.

    Args:
        url (str): URL to download from
        output_path (Path): Path to save the file to
        description (str, optional): Description for the progress bar. Defaults to None.

    Returns:
        Path: Path to the downloaded file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get file size for progress bar
    total_size = int(response.headers.get("content-length", 0))
    desc = description or f"Downloading {output_path.name}"

    with open(output_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return output_path


def extract_tar(tar_path, extract_dir, description=None):
    """
    Extract a tar file with a progress bar.

    Args:
        tar_path (Path): Path to the tar file
        extract_dir (Path): Directory to extract to
        description (str, optional): Description for the progress bar. Defaults to None.

    Returns:
        Path: Path to the extraction directory
    """
    desc = description or f"Extracting {tar_path.name}"
    print(f"{desc}...")

    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting files"):
            tar.extract(member, path=extract_dir)

    return extract_dir


def ensure_dir(path):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path (str or Path): Path to the directory

    Returns:
        Path: Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def download_data(
    task_name: str,
    dataset: str,
    emg_data_dir: str,
    cleanup: bool = True,
) -> Path:
    """
    Download data for a specific task and dataset.

    Args:
        task_name (str): Name of the task (handwriting, discrete_gestures, or wrist)
        dataset (str): Dataset type (full_data or small_subset)
        emg_data_dir (str): Directory where the data will be stored
        cleanup (bool): If True, remove the downloaded tar file after extraction

    Returns:
        Path: Path to the directory where data was downloaded
    """
    base_url = "https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data"

    print(f"Downloading the {dataset} data for {task_name}...")

    # Create directory
    data_dir = ensure_dir(Path(emg_data_dir).expanduser())

    # Check flag file (has data already been downloaded?)
    flag_file = Path(data_dir / f".data_downloaded_{dataset}_{task_name}")
    if flag_file.exists():
        modification_timestamp = flag_file.stat().st_mtime
        last_modified_datetime = datetime.fromtimestamp(modification_timestamp)
        print(
            f"Found data downloaded for {task_name=}, {dataset=} at {data_dir=}. "
            f"Last modified: {last_modified_datetime}. "
            "Assuming data already downloaded!"
        )
        return data_dir

    # Download the tar file
    tar_filename = f"{task_name}_{dataset}.tar"
    tar_path = data_dir / tar_filename
    data_url = f"{base_url}/data/{task_name}/{dataset}.tar"
    download_file(data_url, tar_path, f"Downloading {dataset} data for {task_name}")

    # Extract the tar file
    extract_tar(tar_path, data_dir, "Unzipping the data")

    # Remove the tar file if cleanup is requested
    if cleanup:
        print(f"Deleting the tar file after extraction: {tar_path.name}")
        tar_path.unlink()

    # Download the corpus CSV
    print("Downloading the corpus CSV file...")
    corpus_filename = f"{task_name}_corpus.csv"
    corpus_path = data_dir / corpus_filename
    corpus_url = f"{base_url}/data/{task_name}/{corpus_filename}"
    download_file(corpus_url, corpus_path, f"Downloading {task_name} corpus CSV file")

    # Touch the flag file to signal data was downloaded
    flag_file.touch()

    print(
        f"Data for {task_name} ({dataset}) downloaded and "
        f"extracted to {data_dir.absolute()}"
    )

    return data_dir
# -----------------------------------------------------------

def download_dataset(dataset_dir):
    download_data(
        task_name="wrist",
        dataset="full_data",
        emg_data_dir=dataset_dir,
        cleanup=True,
    )


def check_filename(filename):
    # Example filename: "wrist_user_001_dataset_001.hdf5"
    parts = filename.split('_')
    if len(parts) != 5:
        return False
    if parts[0] != "wrist" or parts[1] != "user" or parts[3] != "dataset":
        return False
    return True

def parse_filename(filename):
    # Example filename: "wrist_user_001_dataset_001.hdf5"
    parts = filename.split('_')
    subj_num = int(parts[2])  # Extract user number
    run_num = int(parts[4])   # Extract dataset number
    return subj_num, run_num


def get_subject_metadata(subj_num):

    if subj_num < 70:
        subj_split = 'train'
    elif subj_num < 80:
        subj_split = 'val'
    else:
        subj_split = 'test'

    return {
        "participant_id": f"sub-{subj_num:03d}",
        "inter_subject_split": subj_split,
        "sex": "u",
        "age": -1,
        "handedness": "u",
    }


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


def load_corpus(dataset_dir: Path) -> pd.DataFrame:
    """Load the corpus CSV defining train/val/test segments."""
    corpus_path = dataset_dir / "wrist_corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus CSV not found at {corpus_path}")
    df = pd.read_csv(corpus_path)
    # Normalise split names to match our convention
    df['split'] = df['split'].str.lower().str.strip()
    return df


def timestamps_to_indices(timestamps: np.ndarray, start_time: float,
                           end_time: float) -> tuple:
    """
    Convert absolute timestamps to array indices.

    Args:
        timestamps: (T,) array of absolute timestamps
        start_time: segment start timestamp
        end_time:   segment end timestamp

    Returns:
        (start_idx, end_idx)
    """
    start_idx = np.searchsorted(timestamps, start_time, side='left')
    end_idx   = np.searchsorted(timestamps, end_time,   side='right')
    return int(start_idx), int(end_idx)


def get_next_run_id(session_group):
    """Generate a new run ID based on existing runs in the session group."""
    existing_runs = [name for name in session_group.keys() if name.startswith("run-")]
    if not existing_runs:
        return "run-001"
    else:
        max_run_num = max(int(name.split('-')[1]) for name in existing_runs)
        return f"run-{max_run_num + 1:03d}"


def get_stage_for_segment(seg_start: float, seg_end: float,
                           stages: pd.DataFrame) -> str:
    """
    Find which stage a corpus segment falls within using midpoint matching.
    Falls back to maximum overlap if midpoint doesn't match any stage.

    Args:
        seg_start: segment start timestamp
        seg_end:   segment end timestamp
        stages:    DataFrame with columns [start, end, name]

    Returns:
        stage name string, or "unknown" if no match found
    """
    seg_mid = (seg_start + seg_end) / 2.0

    # Primary: find stage containing the midpoint
    for _, stage in stages.iterrows():
        if stage["start"] <= seg_mid <= stage["end"]:
            return stage["name"]

    # Fallback: find stage with maximum overlap
    best_overlap = 0.0
    best_name    = "unknown"
    for _, stage in stages.iterrows():
        overlap = min(seg_end, stage["end"]) - max(seg_start, stage["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_name    = stage["name"]

    return best_name


def convert_dataset(dataset_dir, output_fpath, segment_gestures=True):

    # Load corpus — defines segments and their splits
    corpus = load_corpus(dataset_dir)
    # Index by dataset filename for fast lookup
    corpus_by_file = corpus.groupby('dataset')

    # Stage map built dynamically — assigned IDs in order of first encounter
    stage_map = {}

    def get_or_assign_stage_id(stage_name: str) -> int:
        if stage_name not in stage_map:
            stage_map[stage_name] = len(stage_map)
        return stage_map[stage_name]

    # Open output HDF5 file for writing
    with h5py.File(output_fpath, 'w') as f:
        # Create a group for the dataset
        root = f.create_group('neuromotor_wrist')

        # Root-level metadata
        root.attrs["dataset_name"]  = "neuromotor_wrist"
        root.attrs["description"]   = (
            "Generic Neuromotor Interface Wrist dataset. "
            "Continuous wrist flexion/extension and radial/ulnar deviation "
            "recorded with sEMG and wrist angle sensors."
        )
        root.attrs["n_subjects"]    = 100
        root.attrs["sample_rate"]   = 2000
        root.attrs["citation"]      = (
            "https://github.com/facebookresearch/generic-neuromotor-interface"
        )
        
        h5_files = list(dataset_dir.glob("*.hdf5"))
        print("Found {} HDF5 files in dataset directory.".format(len(h5_files)))
        for h5_file in tqdm(h5_files, desc="Processing HDF5 files"):
            if not check_filename(h5_file.name):
                print(f"Skipping file {h5_file.name} as it does not match expected naming convention.")
                continue

            # Parse info from filename
            subj_num, run_num = parse_filename(h5_file.stem)

            subj_id = f"sub-{subj_num:03d}"
            session_id = f'ses-001'             # Assuming that each run is from an independent session

            # Add new user if not already present
            if not subj_id in root:
                subj_meta = get_subject_metadata(subj_num)
                subj_group = root.create_group(subj_id)
                for key, value in subj_meta.items():
                    subj_group.attrs[key] = value
            else:
                subj_group = root[subj_id]

            if not session_id in subj_group:
                session_group = subj_group.create_group(session_id)
                session_group.attrs['session_id'] = session_id
                session_group.attrs['inter_session_split'] = "n/a"
            else:
                session_group = subj_group[session_id]

            with h5py.File(h5_file, 'r') as f:
                # Read the data from the original HDF5 file
                timeseries = f['data']
                emg_data = timeseries['emg']
                wrist_angles = timeseries['wrist_angles']
                timestamps = timeseries['time']

            if segment_gestures:

                # Get corpus segments for this file
                fname = h5_file.name
                if fname not in corpus_by_file.groups:
                    print(f"Warning: {fname} not found in corpus CSV, skipping.")
                    continue
                file_segments = corpus_by_file.get_group(fname).reset_index(drop=True)


                # Get the pandas DataFrame to get the stage/gesture label of each segment
                # stages = pd.read_hdf(h5_file, "stages")

                try:
                    stages = pd.read_hdf(h5_file, "stages")
                except Exception as e:
                    print(f"Warning: Could not read stages from {fname}: {e}")
                    stages = pd.DataFrame(columns=["start", "end", "name"])

                # One run per corpus segment
                for seg_idx, seg in file_segments.iterrows():
                    start_ts = seg["start"]
                    end_ts = seg["end"]
                    split = seg["split"]

                    start_idx, end_idx = timestamps_to_indices(
                        timestamps, start_ts, end_ts
                    )

                    if end_idx - start_idx < 500:  # skip segments shorter than 0.25s
                        continue


                    # Get stage label for this segment
                    stage_name = get_stage_for_segment(start_ts, end_ts, stages)
                    stage_id   = get_or_assign_stage_id(stage_name)

                    run_id = get_next_run_id(session_group)
                    if run_id in session_group:
                        print(f"Warning: {subj_id}/{session_id}/{run_id} already exists, skipping.")
                        continue

                    run_group = session_group.create_group(run_id)

                    run_group.attrs["run_id"]              = run_id
                    run_group.attrs["intra_session_split"] = split
                    run_group.attrs["origin_path"]         = str(h5_file)
                    run_group.attrs["task_name"]           = "wrist_kinematics"
                    run_group.attrs["stage_name"]          = stage_name
                    run_group.attrs["stage_id"]            = stage_id
                    run_group.attrs["gesture_id"]          = stage_id
                    run_group.attrs["is_labelled"]         = True
                    run_group.attrs["segment_start_ts"]    = start_ts
                    run_group.attrs["segment_end_ts"]      = end_ts

                    emg_seg    = emg_data[start_idx:end_idx]
                    angles_seg = wrist_angles[start_idx:end_idx]

                    write_modality(run_group, "emg", emg_seg, {
                        "sample_rate": 2000,
                        "sensor_type": "sEMG-RD",
                        "units":       "raw",
                    })
                    write_modality(run_group, "jntang", angles_seg, {
                        "sample_rate": 2000,
                        "sensor_type": "motion capture",
                        "units":       "degrees",
                    })
                pass
            else:
                
                run_id = f"run-{run_num:03d}"
                if run_id in session_group:
                    print(f"Warning: {subj_id}/{session_id}/{run_id} already exists, skipping.")
                    continue

                run_group = session_group.create_group(run_id)
                run_group.attrs['run_id'] = run_id
                run_group.attrs["origin_path"] = str(h5_file)
                run_group.attrs["intra_session_split"] = "n/a"

                write_modality(run_group, "emg", emg_data, {
                    "sample_rate": 2000,
                    "sensor_type": "sEMG-RD",
                    "units":       "raw",
                })
                write_modality(run_group, "jntang", wrist_angles, {
                    "sample_rate": 2000,
                    "sensor_type": "motion capture",
                    "units":       "degrees",
                })

    print("Conversion complete. Output saved to:", output_fpath)

def main(dataset_dir=None, output_dir=None, download_data=False, include_rest=False, segment_gestures=True, dataset_name="neuromotor_wrist"):
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

    convert_dataset(dataset_dir, output_fpath, segment_gestures=segment_gestures)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/neuromotor_wrist/")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the converted HDF5 files. Defaults to the dataset directory.")
    parser.add_argument("--download_data", action="store_true", help="Whether to download the Neuromotor Wrist dataset.")
    parser.add_argument("--segment_gestures", action="store_true", help="Whether to segment the dataset into individual gestures based on the corpus CSV and hdf5 stages dataframe.")
    args = parser.parse_args()

    main(args.dataset_dir, args.output_dir, args.download_data, segment_gestures=args.segment_gestures)