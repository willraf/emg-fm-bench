""" Module data_audit.py

Audit an hdf5 dataset by checking the data format, and performing visualisation and data analysis to check the data quality.

Usage:
    $ python scripts/datasets/data_audit.py --fpath data/eeg-emg-wristmotion/eeg-emg-wristmotion.hdf5 --dataset_key eeg-emg-wristmotion --primary_key eeg
    $ python scripts/datasets/data_audit.py --fpath data/epn612/epn612.hdf5 --dataset_key epn612 --primary_key emg
    $ python scripts/datasets/data_audit.py --fpath data/ninapro_db5/ninapro_db5.hdf5 --dataset_key ninapro_db5 --primary_key emg
    $ python scripts/datasets/data_audit.py --fpath data/ninapro_db2/ninapro_db2.hdf5 --dataset_key ninapro_db2 --primary_key emg
    $ python scripts/datasets/data_audit.py --fpath data/uci_emg/uci_emg.hdf5 --dataset_key uci_emg --primary_key emg
    $  

Authors:
    Will Raftery

Date:
    17/05/2026
"""

import argparse

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns

def visualise_biosignal(data: np.ndarray, fig_path: str = None):
    """Visualise the biosignal data.
    
    Args:
        data (np.ndarray): Shape (n_samples, n_channels)
        fig_path (str): Path to save the figure. If None, the figure will be displayed.
    """
    offset_amount = np.max(np.abs(data)) * 2

    plt.figure(figsize=(12, 4))
    offsets = np.arange(data.shape[1]) * offset_amount  # Offset each channel for better visualization
    plot_data = data + offsets[np.newaxis, :]  # Add offsets to each channel
    plt.plot(plot_data)
    plt.title("Biosignal Visualization")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid()
    fpath = Path(fig_path) / "biosignal_visualisation.png" if fig_path else None

    if fpath:
        plt.savefig(fpath)
        print(f"Figure saved to {fig_path}")
    else:
        plt.show()


def plot_example(fpath: str, dataset_key: str, primary_key: str, fig_path: str = None):
    """Plot an example of the biosignal data.
    
    Args:
        fpath (str): Path to the hdf5 file.
        dataset_key (str): Key of the dataset to plot.
        primary_key (str): Key of the primary data modality.
        fig_path (str): Path to save the figure. If None, the figure will be displayed.
    """
    with h5py.File(fpath, 'r') as f:
        root = f[dataset_key]

        subject_keys = list(root.keys())
        # Refine list to only include sub-<label> keys
        subject_keys = [key for key in subject_keys if re.match(r'sub-[a-zA-Z0-9]+', key)]
        if not subject_keys:
            print("No subject keys found in the dataset.")
            return
        
        # Get random subject
        subject_key = np.random.choice(subject_keys)
        subj_group = root[subject_key]

        session_keys = list(subj_group.keys())
        session_keys = [key for key in session_keys if re.match(r'ses-[a-zA-Z0-9]+', key)]
        if not session_keys:
            print(f"No session keys found for subject {subject_key}.")
            return

        session_key = np.random.choice(session_keys)
        session_group = subj_group[session_key]

        run_keys = list(session_group.keys())
        run_keys = [key for key in run_keys if re.match(r'run-[a-zA-Z0-9]+', key)]
        if not run_keys:
            print(f"No run keys found for subject {subject_key}, session {session_key}.")
            return
        
        run_key = np.random.choice(run_keys)
        run_group = session_group[run_key]

        data = run_group[primary_key][:]
        print(f"Visualising {primary_key} data for subject {subject_key}, session {session_key}, run {run_key}, data shape: {data.shape}...")
        visualise_biosignal(data, fig_path)


def validate_attrs(attrs_spec, attrs, level_name, require_level, fail_mode='raise'):
    for attr_name, attr_check in attrs_spec.items():
        if attr_name not in attrs:
            message = f"{require_level.capitalize()} {level_name} level attribute '{attr_name}' is missing."
            if fail_mode == 'raise':
                raise ValueError(message)
            elif fail_mode == 'warn':
                print("WARNING:", message)
            continue

        if isinstance(attr_check, type):
            if not isinstance(attrs[attr_name], attr_check):
                message = f"{require_level.capitalize()} {level_name} level attribute '{attr_name}' has type {type(attrs[attr_name])}, expected {attr_check}."
                if fail_mode == 'raise':
                    raise TypeError(message)
                elif fail_mode == 'warn':
                    print("WARNING:", message)
        elif isinstance(attr_check, list):
            if attrs[attr_name] not in attr_check:
                message = f"{require_level.capitalize()} {level_name} level attribute '{attr_name}' has value {attrs[attr_name]}, expected one of {attr_check}."
                if fail_mode == 'raise':
                    raise ValueError(message)
                elif fail_mode == 'warn':
                    print("WARNING:", message)
        elif isinstance(attr_check, tuple):
            if not isinstance(attrs[attr_name], attr_check):
                message = f"{require_level.capitalize()} {level_name} level attribute '{attr_name}' has type {type(attrs[attr_name])}, expected one of {attr_check}."
                if fail_mode == 'raise':
                    raise TypeError(message)
                elif fail_mode == 'warn':
                    print("WARNING:", message)
        else:
            print(f"Invalid check for {require_level.capitalize()} {level_name} level attribute '{attr_name}'.")


def validate_subject_attrs(attrs):
    # Define required attributes and their expected types or allowed values
    required_attrs = {
        'participant_id': str,
    }

    validate_attrs(required_attrs, attrs, 'subject', 'required', fail_mode='raise')

    recommended_attrs = {
        'age': (int, float, np.integer, np.floating),
        'sex': ['m', 'f', 'o', 'u'],
        'handedness': ['r', 'l', 'a', 'u'],
        'origin_path': str,
        'inter_subject_split': ['train', 'val', 'test', 'unlabelled', 'n/a']
    }

    validate_attrs(recommended_attrs, attrs, 'subject', 'recommended', fail_mode='warn')


def validate_session_attrs(attrs):
    required_attrs = {
        'session_id': str,
    }

    validate_attrs(required_attrs, attrs, 'session', 'required', fail_mode='raise')


    recommended_attrs = {
        'inter_session_split': ['train', 'val', 'test', 'unlabelled', 'n/a'], 
        'side': ['l', 'r', 'b', 'u', 'n/a'],
        'origin_path': str,
    }

    validate_attrs(recommended_attrs, attrs, 'session', 'recommended', fail_mode='warn')

def validate_run_attrs(attrs):
    required_attrs = {
    }

    validate_attrs(required_attrs, attrs, 'run', 'required', fail_mode='raise')

    recommended_attrs = {
        'run_id': str,
        'intra_session_split': ['train', 'val', 'test', 'unlabelled', 'n/a'], 
        'origin_path': str,
    }

    validate_attrs(recommended_attrs, attrs, 'run', 'recommended', fail_mode='warn')


def validate_modality_attrs(attrs):
    required_attrs = {
        'sample_rate': (int, np.integer, float, np.floating),
        'n_samples': (int, np.integer),
        'n_channels': (int, np.integer),
    }

    validate_attrs(required_attrs, attrs, 'modality', 'required', fail_mode='raise')

    recommended_attrs = {
        'sensor_type': str,
        'units': str,
        'recording_duration': (int, float, np.integer, np.floating),
    }

    validate_attrs(recommended_attrs, attrs, 'modality', 'recommended', fail_mode='warn')


def validate_dataset_format(fpath: str, dataset_key: str, primary_key: str):

    with h5py.File(fpath, 'r') as f:
        root = f[dataset_key]

        subject_keys = list(root.keys())
        # Refine list to only include sub-<label> keys
        subject_keys = [key for key in subject_keys if re.match(r'sub-[a-zA-Z0-9]+', key)]
        if not subject_keys:
            raise ValueError("No subject keys found in the dataset. Ensure subject groups are named in the format 'sub-<label>'.")
        
        for subj_key in subject_keys:
            subj_group = root[subj_key]

            attrs = dict(subj_group.attrs.items())
            validate_subject_attrs(attrs)

            session_keys = list(subj_group.keys())
            session_keys = [key for key in session_keys if re.match(r'ses-[a-zA-Z0-9]+', key)]
            if not session_keys:
                raise ValueError(f"No session keys found for subject {subj_key}. Ensure session groups are named in the format 'ses-<label>'.")
            
            for ses_key in session_keys:
                ses_group = subj_group[ses_key]

                attrs = dict(ses_group.attrs.items())
                validate_session_attrs(attrs)

                run_keys = list(ses_group.keys())
                run_keys = [key for key in run_keys if re.match(r'run-[a-zA-Z0-9]+', key)]
                if not run_keys:
                    raise ValueError(f"No run keys found for subject {subj_key}, session {ses_key}. Ensure run groups are named in the format 'run-<label>'.")
                
                for run_key in run_keys:
                    run_group = ses_group[run_key]

                    attrs = dict(run_group.attrs.items())
                    validate_run_attrs(attrs)

                    if primary_key not in run_group:
                        print(f"WARNING: Primary key '{primary_key}' not found in subject {subj_key}, session {ses_key}, run {run_key}.")
                        continue

                    attrs = dict(run_group[primary_key].attrs.items())
                    validate_modality_attrs(attrs)

                    data_shape = run_group[primary_key].shape
                    T, C = data_shape
                    if T <= 0 or C <= 0:
                        print(f"WARNING: Primary data for subject {subj_key}, session {ses_key}, run {run_key} has invalid shape {data_shape}.")
                    if T < C:
                        print(f"WARNING: Primary data for subject {subj_key}, session {ses_key}, run {run_key} has more channels ({C}) than time points ({T}). Check if the data is transposed.")
        
        print(f"Found {len(subject_keys)} subjects: {subject_keys}")


def extract_dataset_metadata(hdf5_path, root_key, data_keys=['emg']):
    """
    Crawls a unified biosignal HDF5 file and flattens metadata into a DataFrame.
    """
    records = []
    with h5py.File(hdf5_path, 'r') as f:
        if root_key not in f:
            raise KeyError(f"Root key '{root_key}' not found in {hdf5_path}")
        
        root = f[root_key]
        for subj_id in tqdm(root.keys(), desc=f"Scanning {root_key}"):
            subj_group = root[subj_id]
            subj_meta = dict(subj_group.attrs)
            
            for sess_id in subj_group.keys():
                if not isinstance(subj_group[sess_id], h5py.Group):
                    continue
                
                sess_group = subj_group[sess_id]
                sess_meta = dict(sess_group.attrs)

                for run_key in sess_group.keys():
                    if not isinstance(sess_group[run_key], h5py.Group):
                        continue
                    
                    run_group = sess_group[run_key]
                    run_meta = dict(run_group.attrs)

                    record = {
                        **subj_meta,
                        **sess_meta,
                        **run_meta,
                        'subj_key': subj_id,
                        'sess_key': sess_id,
                        'run_key': run_key
                    }

                    for data_key in data_keys:
                        if data_key not in run_group:
                            print(f"Warning: '{data_key}' not found in {subj_id}/{sess_id}/{run_key}")
                            continue
                        data = run_group[data_key]
                        if not isinstance(data, h5py.Dataset):
                            print(f"Warning: '{data_key}' in {subj_id}/{sess_id}/{run_key} is not a dataset")
                            continue
                            
                        t, c = data.shape
                        fs = data.attrs.get('sample_rate', None)
                        if fs is None:
                            print(f"Warning: 'sample_rate' attribute not found for '{data_key}' in {subj_id}/{sess_id}/{run_key}")
                            continue

                        record.update({
                            f'{data_key}_num_samples': t,
                            f'{data_key}_num_channels': c,
                            f'{data_key}_duration_sec': t / fs,
                            f'{data_key}_mean': np.mean(data),
                            f'{data_key}_std': np.std(data)
                        })

                    records.append(record)
                
    df = pd.DataFrame(records)
    # Ensure participant_id is a sortable type (int if possible, else string)
    # Ensure participant_num is a sortable type (int if possible)
    # Participarn ID is often a string like "subject_01", so we can extract the numeric part for sorting
    try:
        regex = re.compile(r'\d+')
        df['participant_number'] = df['participant_id'].apply(lambda x: int(regex.search(str(x)).group()) if regex.search(str(x)) else x)
        df.sort_values('participant_number', inplace=True)
    except:
        pass
        
    return df


class BiosignalAuditor:
    def __init__(self, df, dataset_name="Dataset", label_key='gesture_id', data_key='{self.data_key}', fig_path=None):
        self.df = df
        self.name = dataset_name.upper()
        # Filter out unlabelled for core distribution plots
        self.df_clean = df[df[label_key] != -1].copy()
        self.label_key = label_key
        self.data_key = data_key
        self.fig_path = fig_path
        
        plt.style.use('seaborn-v0_8-muted')
        sns.set_context("talk")

    def run_full_audit(self):
        """Executes all visualization and statistical checks."""
        self.plot_overview()
        self.plot_length_leakage_check()
        self.print_summary()

    def plot_overview(self):
        fig, axes = plt.subplots(5, 2, figsize=(40, 50))
        plt.subplots_adjust(hspace=0.4)
            
        # 1. Class Distribution (Gesture ID)
        # This checks for "Label Imbalance"
        sns.countplot(data=self.df, x=self.label_key, hue='intra_session_split', ax=axes[0,0])
        axes[0,0].set_title("Gesture Class Distribution (Intra-Session Split)")
        axes[0,0].tick_params(axis='x', rotation=45)


        sns.countplot(data=self.df, x=self.label_key, hue='inter_subject_split', ax=axes[0,1])
        axes[0,1].set_title("Gesture Class Distribution (Inter-Subject Split)")
        axes[0,1].tick_params(axis='x', rotation=45)


        # 2. Duration Distribution
        sns.histplot(data=self.df_clean, x=f'{self.data_key}_duration_sec', hue='intra_session_split', multiple="stack", ax=axes[1,0])
        axes[1,0].set_title("Distribution of Trial Durations - Intra-session split (Seconds)")

        sns.histplot(data=self.df_clean, x=f'{self.data_key}_duration_sec', hue='inter_subject_split', multiple="stack", ax=axes[1,1])
        axes[1,1].set_title("Distribution of Trial Durations - Inter-subject split (Seconds)")

        # 3. Leakage Heatmap (Subject-to-Split Mapping)
        leak_table = pd.crosstab(self.df['participant_id'], self.df['intra_session_split'])
        sns.heatmap(leak_table, annot=True, cbar=False, cmap='Blues', ax=axes[2,0])
        axes[2,0].set_title("Session Leakage Check")

        leak_table = pd.crosstab(self.df['participant_id'], self.df['inter_subject_split'])
        sns.heatmap(leak_table, annot=True, cbar=False, cmap='Blues', ax=axes[2,1])
        axes[2,1].set_title("Subject Leakage Check")


        # 4. Signal Quality (Mean vs Std per Session)
        sns.scatterplot(data=self.df_clean, x=f'{self.data_key}_mean', y=f'{self.data_key}_std', hue='intra_session_split', alpha=0.5, ax=axes[3,0])
        axes[3,0].set_title("Signal Energy (Mean vs Std) - Quality Check")
        # axes[3,0].get_legend().remove()


        # 5. Signal Quality (Mean vs Std per Subject)
        sns.scatterplot(data=self.df_clean, x=f'{self.data_key}_mean', y=f'{self.data_key}_std', hue='participant_id', alpha=0.5, ax=axes[3,1])
        axes[3,1].set_title("Signal Energy (Mean vs Std) - Quality Check")
        # axes[3,1].get_legend().remove()


        # 6. Sex Distribution pie chart
        if 'sex' in self.df_clean.columns:
            axes[4,0].pie(self.df_clean['sex'].value_counts(), labels=self.df_clean['sex'].unique(), autopct='%1.1f%%', colors=sns.color_palette('Set2')) 
            axes[4,0].set_title("Gender Distribution")

        # 7. Handedness Distribution
        if 'handedness' in self.df_clean.columns:
            axes[4,1].pie(self.df_clean['handedness'].value_counts(), labels=self.df_clean['handedness'].unique(), autopct='%1.1f%%', colors=sns.color_palette('Set2'))
            axes[4,1].set_title("Handedness Distribution")

        if self.fig_path:
            fpath = Path(self.fig_path) / "dataset_overview.png"
            plt.savefig(fpath)
            print(f"Overview figure saved to {fpath}")
        else:   
            plt.show()

    def plot_length_leakage_check(self):
        """Checks if gesture ID is perfectly predictable by signal duration."""
        plt.figure(figsize=(16, 6))
        sns.boxplot(data=self.df_clean.sort_values(self.label_key), x=self.label_key, y=f'{self.data_key}_duration_sec', palette='husl')
        plt.axhline(self.df_clean[f'{self.data_key}_duration_sec'].mean(), color='red', linestyle='--', alpha=0.6, label='Global Mean')
        plt.title(f"{self.name}: Signal Length Leakage Audit")
        plt.xticks(rotation=90 if self.df_clean[self.label_key].nunique() > 10 else 0)
        plt.legend()

        if self.fig_path:
            fpath = Path(self.fig_path) / "length_leakage_check.png"
            plt.savefig(fpath)
            print(f"Length leakage figure saved to {fpath}")
        else:
            plt.show()

    def print_summary(self):
        print(f"\n{'='*40}\n{self.name} AUDIT REPORT\n{'='*40}")
        print(f"Total Subjects: {self.df['participant_id'].nunique()}")
        print(f"Total Trials:   {len(self.df)}")
        print(f"Total {self.data_key} Time: {self.df[f'{self.data_key}_duration_sec'].sum() / 3600:.2f} hours")
        
        for col in ['intra_session_split', 'inter_session_split', 'subject_split', 'inter_subject_split']:
            if col in self.df.columns:
                print(f"\nSplit Distribution ({col}):")
                print(self.df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

        # Check for Inter-Subject Leakage
        subj_split_col = next((c for c in ['subject_split', 'inter_subject_split'] if c in self.df.columns), None)
        if subj_split_col:
            leak_check = self.df.groupby('participant_id')[subj_split_col].nunique()
            leaked = leak_check[leak_check > 1].index.tolist()
            if leaked:
                print(f"\n⚠️ WARNING: {len(leaked)} subjects found in multiple splits: {leaked[:5]}...")
            else:
                print("\n✅ Clean: No subject leakage detected across splits.")


def plot_global_amplitude_histogram(hdf5_path, root_key, n_runs_to_sample=50, primary_key='emg'):
    """
    Plots a histogram of all raw EMG datapoints across a subset of runs.
    This reveals the statistical distribution (PDF) of the signal.
    """
    all_values = []
    n = 0
    with h5py.File(hdf5_path, 'r') as f:
        for subj_id in f[root_key].keys():
            subj_group = f[root_key][subj_id]
            for sess_id in subj_group.keys():
                sess_group = subj_group[sess_id]
                for run_key in sess_group.keys():
                    run_group = sess_group[run_key]
                    emg_data = run_group[primary_key][:]
                    all_values.extend(emg_data.flatten())
                    n += 1
                    if n >= n_runs_to_sample:
                        break
    
    all_values = np.array(all_values)

    plt.figure(figsize=(12, 6))
    
    # Using a log scale for the Y-axis is often better for EMG 
    # because the 'Rest' zeros dominate the count.
    sns.histplot(all_values, bins=100, kde=True, color='teal')
    
    plt.yscale('log') 
    plt.title(f"{root_key}: Global Amplitude Distribution (All Datapoints)")
    plt.xlabel("Amplitude (Normalized units)")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add vertical lines for standard deviations
    std = np.std(all_values)
    plt.axvline(std, color='orange', linestyle='--', label=f'1 Std ({std:.2f})')
    plt.axvline(-std, color='orange', linestyle='--')
    plt.legend()
    


def main(fpath: str, dataset_key: str, primary_key: str, label_key: str = 'gesture_id'):
    print("Auditing dataset...")
    fig_path = Path(fpath).parent / "audit_figures"
    fig_path.mkdir(exist_ok=True)

    validate_dataset_format(fpath, dataset_key, primary_key)

    plot_example(fpath, dataset_key, primary_key, fig_path=fig_path)

    df = extract_dataset_metadata(fpath, dataset_key, data_keys=[primary_key])

    auditor = BiosignalAuditor(df, dataset_name=dataset_key, label_key=label_key, data_key=primary_key, fig_path=fig_path)
    auditor.run_full_audit()

    plot_global_amplitude_histogram(fpath, dataset_key, n_runs_to_sample=50, primary_key=primary_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert BIDS dataset to HDF5')
    parser.add_argument('--fpath', type=str, required=True, help='Path to the HDF5 file to audit')
    parser.add_argument('--dataset_key', type=str, default='dataset', help='Key of the dataset to audit')
    parser.add_argument('--primary_key', type=str, default='eeg', help='Key of the primary data modality to visualise')
    parser.add_argument('--label_key', type=str, default='gesture_id', help='Key of the label column')
    args = parser.parse_args()

    main(args.fpath, args.dataset_key, args.primary_key, args.label_key)