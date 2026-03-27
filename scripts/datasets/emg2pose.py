import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import argparse
import urllib.request
import tarfile
from pathlib import Path
from tqdm import tqdm


def convert_emg2pose(src_dir, dest_h5_path):
    src_path = Path(src_dir)
    h5_files = list(src_path.glob("*.hdf5"))
    # Exclude destimation h5 file
    h5_files = [f for f in h5_files if f.name != dest_h5_path.name]
    
    with h5py.File(dest_h5_path, "w") as out_f:
        root = out_f.create_group("emg2pose")
        
        for fpath in tqdm(h5_files, desc="Converting emg2pose"):
            with h5py.File(fpath, "r") as src_f:
                # 1. Extract Metadata
                attrs = src_f["emg2pose"].attrs
                if 'user' not in attrs or 'session' not in attrs:
                    print(f"Skipping {fpath.name} due to missing 'user' or 'session' attributes.")
                    continue
                subj_id = f"sub-{attrs['user']}"
                ses_id = f"ses-{attrs['session']}"
                run_id = f"run-{fpath.stem}" # Unique name per file

                if attrs['generalization'] in ['user', 'user_stage']:
                    subj_split = attrs['split']
                else:
                    subj_split = 'unknown'

                if attrs['generalization'] in ['stage', 'user_stage']:
                    task_split = attrs['split']
                else:
                    task_split = 'unknown'

                if subj_id not in root:
                    subj_group = root.create_group(subj_id)
                    subj_group.attrs['participant_id'] = subj_id
                    subj_group.attrs['inter_subject_split'] = subj_split
                else:
                    subj_group = root[subj_id]
                    subj_split = subj_group.attrs['inter_subject_split']
                    if attrs['generalization'] in ['user', 'user_stage']:
                        assert subj_split == attrs['split'], f"Mismatch in inter_subject_split for {subj_id}: existing {subj_split} vs new {attrs['split']}"
                

                subj_group.attrs['inter_subject_split'] = subj_split

                if ses_id not in subj_group:
                    ses_group = subj_group.create_group(ses_id)
                    ses_group.attrs['session_id'] = ses_id
                    ses_group.attrs['inter_subject_split'] = subj_split
                    ses_group.attrs['participant_id'] = subj_id
                else:
                    ses_group = subj_group[ses_id]
                    assert ses_group.attrs['inter_subject_split'] == subj_split, f"Mismatch in inter_subject_split for {ses_id}: existing {ses_group['inter_subject_split']} vs new {subj_split}"

                run_grp = ses_group.create_group(run_id)

                # 3. Transfer and Normalize Data
                ts = src_f["emg2pose"]["timeseries"]
                
                # EMG (16 channels) - Check if scaling is needed!
                # If these are raw 2000Hz signals, they might be in microvolts.
                # Inspect a sample before deciding on the divisor.
                emg = ts['emg'][:].astype(np.float32) 
                
                # Joint Angles (20 DoF)
                pose = ts['joint_angles'][:].astype(np.float32)

                time = ts['time'][:].astype(np.float32)

                # Create Datasets
                run_grp.create_dataset("time", data=time, compression="gzip", compression_opts=4)
                run_grp.create_dataset("emg", data=emg, compression="gzip", compression_opts=4)
                run_grp.create_dataset("kinematics", data=pose, compression="gzip", compression_opts=4)

                run_grp['emg'].attrs['sample_rate'] = attrs['sample_rate']
                run_grp['kinematics'].attrs['sample_rate'] = attrs['sample_rate']
                
                # 4. Copy specific attributes to Run level
                run_grp.attrs['sample_rate'] = attrs['sample_rate']
                run_grp.attrs['side'] = attrs['side']
                run_grp.attrs['stage'] = attrs['stage']
                run_grp.attrs['moving_hand'] = attrs['moving_hand']

                run_grp.attrs['inter_subject_split'] = subj_split
                run_grp.attrs['inter_task_split'] = task_split
                run_grp.attrs['original_file'] = fpath.name
                # assert fpath.name == attrs['filename'], f"Filename attribute mismatch: {fpath.name} vs {attrs['filename']}"

                run_grp.attrs['generalization'] = attrs['generalization']
                run_grp.attrs['held_out_user'] = attrs['held_out_user']
                run_grp.attrs['held_out_stage'] = attrs['held_out_stage']
                run_grp.attrs['split'] = attrs['split']


def download_dataset(dataset_dir, mini=True):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if mini:
        fname = "emg2pose_dataset_mini.tar"
        url = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"
        tar_path = dataset_dir / fname
    else:
        fname = "emg2pose_dataset.tar"
        url = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
        tar_path = dataset_dir / fname

    print(f"Downloading {fname}...")
    try:
        urllib.request.urlretrieve(url, tar_path)
        
        # 2. Extract and Flatten
        print("Extracting and cleaning up directory structure...")
        with tarfile.open(tar_path, 'r') as tar:
            # Extract all files while flattening the directory structure
            for member in tar.getmembers():
                if member.isfile():
                    # Extract to the dataset_dir with a flat structure
                    member.name = Path(member.name).name  # Flatten the path
                    tar.extract(member, path=dataset_dir)
            
        # Cleanup
        tar_path.unlink()
        print(f"Successfully downloaded and extracted emg2pose in {dataset_dir}")
        
    except Exception as e:
        print(f"An error occurred during emg2pose download: {e}")
        if tar_path.exists():
            tar_path.unlink()





def main(dataset_dir, output_dir=None, download_data=False, mini=True):

    dataset_dir = Path(dataset_dir)

    if output_dir is None:
        output_fpath = dataset_dir / "emg2pose.hdf5"
    else:
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_fpath = output_dir / "emg2pose.hdf5"

    if download_data:
        download_dataset(dataset_dir)

    convert_emg2pose(dataset_dir, output_fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/emg2pose/")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the converted HDF5 files. Defaults to the dataset directory.")
    parser.add_argument("--download_data", action="store_true", help="Whether to download the emg2pose dataset.")
    parser.add_argument("--mini", action="store_true", help="Whether to download the mini version of the dataset (for testing).")
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.download_data)