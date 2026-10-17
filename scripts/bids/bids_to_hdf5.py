""" Module bids_to_hdf5.py

Universal converter from BIDS format to the Unified Biosignal HDF5 Format (v1.0).

Supports modalities: EMG, EEG, Motion (ACCEL, GYRO, ORNT/quaternion, JNTANG)
Enforces: subject --> session --> run --> modality hierarchy

Usage:
    DEPRICATED:
    python -m scripts.bids.bids_to_hdf5 \
        --bids_dir data/ninapro_db5_bids/ \
        --output_path data/ninapro_db5_bids/ninapro_db5.hdf5 \
        --dataset_name ninapro_db5

    $ python scripts/bids/bids_to_hdf5.py --config scripts/bids/bids_to_hdf5.yaml 

Date:
    11/05/2026
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyedflib
import re

from dataclasses import dataclass, field, InitVar
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Constants — metadata keys to carry through to HDF5
# ---------------------------------------------------------------------------

# participants.tsv column → HDF5 subject attr key
PARTICIPANT_ATTR_MAP = {
    'age':                   'age',
    'sex':                   'sex',
    'handedness':            'handedness',
    'weight':                'weight',
    'height':                'height',
    'forearm_circumference': 'forearm_circumference',
}

# BIDS sidecar JSON key → HDF5 run/modality attr key
SIDECAR_ATTR_MAP = {
    'SamplingFrequency':  'sample_rate',
    'TaskName':           'task_name',
    'Manufacturer':       'manufacturer',
    'ManufacturersModelName': 'sensor_type',
}

# Modality suffixes this script handles
HANDLED_SUFFIXES = {'emg', 'eeg', 'motion'}


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_participants(bids_dir: Path) -> dict:
    """
    Parse participants.tsv into a dict keyed by participant_id.
    Returns {} if file doesn't exist.
    """
    tsv_path = bids_dir / 'participants.tsv'
    if not tsv_path.exists():
        logging.warning("No participants.tsv found — subject metadata will be empty.")
        return {}

    df = pd.read_csv(tsv_path, sep='\t', dtype=str)     # read as str to avoid int/float coercion
    df = df.where(pd.notna(df), other='n/a')            # replace NaN with 'n/a'
    return {row['participant_id']: row.to_dict() for _, row in df.iterrows()}


def load_sidecar_json(path: Path) -> dict:
    """Load a BIDS sidecar JSON, return {} if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_events_tsv(path: Path) -> pd.DataFrame | None:
    """Load a BIDS events.tsv, return None if missing."""
    if path.exists():
        return pd.read_csv(path, sep='\t')
    print(f"Warning: No events.tsv found at {path} — skipping event parsing for this run.")
    return None


def load_channels_tsv(path: Path) -> pd.DataFrame | None:
    """Load a BIDS channels.tsv, return None if missing."""
    if path is None:
        return None
    if path.exists():
        return pd.read_csv(path, sep='\t')
    return None


def resolve_sidecar(data_dir: Path, data_entities: dict) -> Path | None:
    """
    Resolve the most specific sidecar JSON file for a given data file based on BIDS inheritance.

    Args:
        dir: Directory to search for sidecar JSONs (e.g. subject/session/modality folder)
        data_entities: Dict of BIDS entities parsed from the data file name (e.g. {'sub': '01', 'ses': '01', 'task': 'ExerciseB'})
    """
    suffix = data_entities.get('suffix', 'unknown')
    sidecar_fpath = None

    while sidecar_fpath is None:
        sidecar_fpath = get_inheritance_fpath(data_dir, data_entities, suffix=suffix, ext='.json')
        if (data_dir / 'participants.tsv').exists():
            break
        data_dir = data_dir.parent
        # Check if we've reached the root of the BIDS dataset by looking for participants.tsv


    if sidecar_fpath is None:
        raise FileNotFoundError(f"No sidecar JSON found for {data_entities} in {data_dir} or any parent directories")

    return load_sidecar_json(sidecar_fpath)


# def resolve_sidecar(bids_dir: Path, subject: str, session: str | None,
#                     suffix: str, entities: dict) -> dict:
#     """
#     Resolve BIDS sidecar JSON with inheritance.
    
#     BIDS inheritance works by walking UP the directory tree from the file,
#     merging JSONs at each level. More specific (closer to file) wins.
    
#     At each directory level, valid sidecar filenames are built from
#     optional entity combinations in this order:
#         [sub]-[ses]-[task]-[acq]-[run]-[tracksys]-[suffix].json
    
#     Inheritance order (least → most specific):
#         root → sub/ → sub/ses/ → sub/ses/modality/ (file-level)
#     """
#     merged = {}

#     def _load_and_merge(path: Path):
#         """Load JSON at path and merge into accumulated dict. 
#         More specific keys already in merged are NOT overwritten."""
#         data = load_sidecar_json(path)
#         if data:
#             logging.debug(f"  Loaded sidecar: {path}")
#         merged.update({k: v for k, v in data.items() if k not in merged})

#     # ------------------------------------------------------------------
#     # Build ordered list of candidate sidecar filenames at a given dir.
#     # BIDS spec: entities are included only if present, in canonical order.
#     # More entities = more specific = checked last = wins on conflict.
#     # ------------------------------------------------------------------
#     def candidate_sidecars(directory: Path, 
#                            include_sub: bool = False,
#                            include_ses: bool = False) -> list[Path]:
#         """
#         Generate candidate sidecar paths at a directory level,
#         ordered from least to most specific.
#         """
#         # Canonical BIDS entity order for sidecar matching
#         # Only include entities that are present in this file's entities dict
#         sub     = entities.get('sub')
#         ses     = entities.get('ses')
#         task    = entities.get('task')
#         acq     = entities.get('acq')
#         run     = entities.get('run')
#         tracksys = entities.get('tracksys')

#         # Build entity combinations from least to most specific
#         # Each tuple is a sequence of (key, value) pairs to include
#         candidates = []

#         # 1. Suffix only — least specific, applies to all files of this type
#         candidates.append([])

#         # 2. Task only
#         if task:
#             candidates.append([('task', task)])

#         # 3. Acq only
#         if acq:
#             candidates.append([('acq', acq)])

#         # 4. Task + acq
#         if task and acq:
#             candidates.append([('task', task), ('acq', acq)])

#         # 5. Task + run
#         if task and run:
#             candidates.append([('task', task), ('run', run)])

#         # 6. Task + tracksys (motion-specific)
#         if task and tracksys:
#             candidates.append([('task', task), ('tracksys', tracksys)])

#         # 7. Task + acq + run
#         if task and acq and run:
#             candidates.append([('task', task), ('acq', acq), ('run', run)])

#         # 8. Task + acq + tracksys
#         if task and acq and tracksys:
#             candidates.append([('task', task), ('acq', acq), ('tracksys', tracksys)])

#         # 9. Full specificity — all present entities
#         full = []
#         if include_sub and sub: full.append(('sub', sub))
#         if include_ses and ses: full.append(('ses', ses))
#         if task:                full.append(('task', task))
#         if acq:                 full.append(('acq', acq))
#         if run:                 full.append(('run', run))
#         if tracksys:            full.append(('tracksys', tracksys))
#         if full not in candidates:
#             candidates.append(full)

#         # Build actual paths
#         paths = []
#         for entity_pairs in candidates:
#             if entity_pairs:
#                 fname = '_'.join(f'{k}-{v}' for k, v in entity_pairs) + f'_{suffix}.json'
#             else:
#                 fname = f'{suffix}.json'
#             paths.append(directory / fname)

#         return paths

#     # ------------------------------------------------------------------
#     # Walk from least to most specific, loading and merging at each level
#     # ------------------------------------------------------------------

#     # 1. Root level — no subject/session prefix
#     for path in candidate_sidecars(bids_dir):
#         _load_and_merge(path)

#     # 2. Subject level — files prefixed with sub-XX
#     sub_dir = bids_dir / subject
#     for path in candidate_sidecars(sub_dir, include_sub=True):
#         _load_and_merge(path)

#     # 3. Session level — files prefixed with sub-XX_ses-XX
#     if session:
#         ses_dir = sub_dir / session
#         for path in candidate_sidecars(ses_dir, include_sub=True, include_ses=True):
#             _load_and_merge(path)

#         # 4. Modality folder level — most specific
#         # BIDS stores data files in modality subfolders (emg/, motion/, eeg/ etc.)
#         modality_dir = ses_dir / suffix
#         for path in candidate_sidecars(modality_dir, include_sub=True, include_ses=True):
#             _load_and_merge(path)

#     else:
#         # No session — modality folder is directly under subject
#         modality_dir = sub_dir / suffix
#         for path in candidate_sidecars(modality_dir, include_sub=True):
#             _load_and_merge(path)

#     return merged


def write_subject_attrs(subject_group: h5py.Group, participant_row: dict,
                        inter_subject_split: str):
    """Write standardised subject-level attrs from participants.tsv row."""
    subject_group.attrs['participant_id'] = subject_group.name.split('/')[-1]
    subject_group.attrs['inter_subject_split'] = inter_subject_split

    for tsv_col, attr_key in PARTICIPANT_ATTR_MAP.items():
        val = participant_row.get(tsv_col, 'n/a')
        if val == 'n/a':
            continue
        # Attempt numeric coercion for age, weight, height etc.
        try:
            val = int(val)
        except (ValueError, TypeError):
            try:
                val = float(val)
            except (ValueError, TypeError):
                pass    # leave as string
        subject_group.attrs[attr_key] = val


def write_sidecar_attrs(group: h5py.Group, sidecar: dict):
    """Write mapped sidecar JSON fields to a group's attrs."""
    for bids_key, hdf5_key in SIDECAR_ATTR_MAP.items():
        if bids_key in sidecar:
            group.attrs[hdf5_key] = sidecar[bids_key]


# ---------------------------------------------------------------------------
# EDF signal reading
# ---------------------------------------------------------------------------

def read_edf(edf_path: Path, channels_tsv: pd.DataFrame | None = None
             ) -> dict[str, np.ndarray]:
    """
    Read an EDF file using. Returns a dict mapping modality group name
    to (T, C) float32 array.

    If channels_tsv is provided, uses the 'type' column to route channels
    into separate arrays. Otherwise puts everything under the file suffix.
    """

    f = pyedflib.EdfReader(str(edf_path))
    n = f.signals_in_file
    ch_names = f.getSignalLabels()
    data = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        data[i, :] = f.readSignal(i)

    data = data.T.astype(np.float32)   # transpose to (T, C) and convert to float32

    if channels_tsv is None:
        # No channel type info — return everything as one array
        suffix = edf_path.stem.split('_')[-1]   # e.g. "emg" from "sub-01_task-X_emg"
        return {suffix: data}

    # Match channel names from edf file to channels.tsv rows to get channel type (e.g. emg, ecg, emg)
    # Route channels by type using channels.tsv
    type_col = channels_tsv['type'].str.upper() if 'type' in channels_tsv.columns else None

    if type_col is None:
        suffix = edf_path.stem.split('_')[-1]
        return {suffix: data}

    result = {}
    for ch_idx, ch_name in enumerate(ch_names):
        # Find this channel in channels.tsv
        if ch_idx >= len(channels_tsv):
            logging.warning(f"Channel index {ch_idx} out of bounds for channels.tsv with {len(channels_tsv)} rows")
            continue
    
        row = channels_tsv[channels_tsv['name'] == ch_name]
        if row.empty:
            ch_type = 'unknown'
            ch_type = channels_tsv['type'].values[ch_idx].upper() if 'type' in channels_tsv.columns else 'unknown'
        else:
            ch_type = row['type'].values[0].upper()

        # Map BIDS type to HDF5 group name
        group_name = ch_type.lower()

        if group_name not in result:
            result[group_name] = []
        result[group_name].append(data[:, ch_idx])

    # Stack channel lists into (T, C) arrays
    return {k: np.stack(v, axis=1) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Motion TSV reading
# ---------------------------------------------------------------------------

def read_motion_tsv(tsv_path: Path, channels_tsv: pd.DataFrame | None
                    ) -> dict[str, np.ndarray]:
    """
    Read a BIDS motion .tsv file and split by channel type.
    Returns dict mapping HDF5 group name → (T, C) float32 array.

    Args:
        tsv_path:     Path to *_motion.tsv
        channels_tsv: Corresponding *_channels.tsv as DataFrame
    """
    # Motion TSV has no header — columns are ordered as in channels.tsv
    data = pd.read_csv(tsv_path, sep='\t', header=None).values.astype(np.float32)  # (T, C)

    if channels_tsv is None:
        logging.warning(f"No channels.tsv for {tsv_path.name} — storing as 'motion'")
        return {'motion': data}

    result = {}
    for ch_idx, row in channels_tsv.iterrows():
        ch_type = str(row.get('type', 'unknown')).upper()
        group_name = ch_type.lower()

        if group_name not in result:
            result[group_name] = []
        result[group_name].append(data[:, ch_idx])

    return {k: np.stack(v, axis=1) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Modality writing
# ---------------------------------------------------------------------------

def write_modality(run_group: h5py.Group, group_name: str,
                   data: np.ndarray, sidecar: dict,
                   origin_path: str):
    """
    Write a single modality array to a run group with standardised attrs.

    Args:
        run_group:   HDF5 run group
        group_name:  e.g. 'emg', 'accel', 'jntang'
        data:        (T, C) float32 array
        sidecar:     Parsed BIDS sidecar JSON for this file
        origin_path: BIDS file path for provenance
    """
    ds = run_group.create_dataset(group_name, data=data, compression='lzf')
    ds.attrs['origin_path'] = origin_path

    # Sample rate — try sidecar first, then run-level attrs
    sample_rate = sidecar.get('SamplingFrequency') or run_group.attrs.get('sample_rate')
    if sample_rate is not None:
        ds.attrs['sample_rate'] = int(sample_rate)

    # Sensor type
    sensor_type = sidecar.get('ManufacturersModelName') or sidecar.get('Manufacturer')
    if sensor_type:
        ds.attrs['sensor_type'] = sensor_type

    # Units
    if 'Units' in sidecar:
        ds.attrs['units'] = sidecar.get('Units')

    ds.attrs['n_samples'] = data.shape[0]
    if len(data.shape) > 1: 
        ds.attrs['n_channels'] = data.shape[1]

    ds.attrs['recording_duration'] = data.shape[0] / sample_rate if sample_rate else 'n/a'


# ---------------------------------------------------------------------------
# Events parsing
# ---------------------------------------------------------------------------

def reconstruct_timeseries_events(events_df: pd.DataFrame,
                                    n_samples: int,
                                    sample_rate: int,
                                    timeseries_cols: set[str],
                                    preferred_trial_type: str = None
                                    ) -> dict[str, np.ndarray]:
    """
    Reconstruct dense (T,) label arrays from a sparse BIDS events DataFrame.

    For each event row, fills samples [sample_onset : sample_onset + sample_duration]
    with the event's label value. Gaps between events are filled with 0 (rest).

    Args:
        events_df:            Parsed events.tsv DataFrame
        n_samples:            Total number of samples in the recording (from signal data)
        sample_rate:          Sampling frequency in Hz
        timeseries_cols:      Set of column names to reconstruct as dense arrays
        trial_type_col:       Name of the trial_type column
        preferred_trial_type: Which trial_type to use ('action' or 'prompt').
                              If not present, falls back to using all events.

    Returns:
        Dict mapping column name → (T,) int32 numpy array
    """
    if events_df is None or events_df.empty:
        return {}

    # Filter to preferred trial type if available
    if 'trial_type' in events_df.columns and preferred_trial_type:
        filtered = events_df[events_df['trial_type'] == preferred_trial_type]
        if filtered.empty:
            logging.warning(f"No '{preferred_trial_type}' events found — using all events")
            filtered = events_df
    else:
        filtered = events_df

    # Only reconstruct columns that exist in the dataframe
    cols_to_reconstruct = timeseries_cols & set(filtered.columns)
    if not cols_to_reconstruct:
        logging.warning(f"None of {timeseries_cols} found in events.tsv columns: {list(filtered.columns)}")
        return {}

    # Determine sample indices — prefer 'sample' column (lossless) over onset/sample_rate
    if 'sample' in filtered.columns and 'sample_duration' in filtered.columns:
        use_sample_cols = True
    elif 'onset' in filtered.columns and 'duration' in filtered.columns:
        use_sample_cols = False
    else:
        logging.warning("Events TSV has neither (sample, sample_duration) nor (onset, duration) columns")
        return {}

    # Initialise timeseries arrays — default 0 (rest)
    timeseries = {col: np.zeros(n_samples, dtype=np.int32) for col in cols_to_reconstruct}

    for _, row in filtered.iterrows():
        # Get sample indices
        if use_sample_cols:
            start = int(row['sample'])
            dur   = int(row['sample_duration'])
        else:
            start = round(float(row['onset'])    * sample_rate)
            dur   = round(float(row['duration']) * sample_rate)

        end = min(start + dur, n_samples)

        if start >= n_samples:
            continue    # event starts after signal ends — skip

        for col in cols_to_reconstruct:
            val = row[col]
            if pd.isna(val):
                continue
            timeseries[col][start:end] = int(val)

    return timeseries

def parse_events(events_df: pd.DataFrame, 
                              run_group: h5py.Group,
                              n_samples: int,
                              sample_rate: int,
                              timeseries_cols: set[str],
                              preferred_trial_type: str = None):
    """
    Extract timeseries events from events_df and write as datasets under run_group.
    """
    if events_df is None:
        return

    timeseries_arrays = reconstruct_timeseries_events(events_df, n_samples, sample_rate, timeseries_cols, preferred_trial_type)

    for col, arr in timeseries_arrays.items():
        if col in run_group:
            logging.warning(f"  Label dataset '{col}' already exists in {run_group.name} — skipping")
            continue

        sidecar = {
            'description': f"Dense label array reconstructed from BIDS events.tsv ({col})",
            'sample_rate': sample_rate,
            'units': 'n/a',
            'n_samples': n_samples,
            'recording_duration': n_samples / sample_rate if sample_rate else 'n/a',
        }

        write_modality(run_group, col, arr, sidecar=sidecar, origin_path=f"{run_group.name}/events.tsv")

        logging.debug(f"  Written dense label: {col} shape={arr.shape} "
                      f"unique={np.unique(arr)}")

# ---------------------------------------------------------------------------
# Core BIDS entity parsing
# ---------------------------------------------------------------------------

def parse_bids_filename(stem: str) -> dict:
    """
    Parse a BIDS filename stem into entity key-value pairs using regex.
    e.g. "sub-01_ses-01_task-ExerciseB_run-01_emg.edf"
    → {'sub': '01', 'ses': '01', 'task': 'ExerciseB', 'run': '01', 'suffix': 'emg', 'ext': '.edf'}
    """
    pattern = re.compile(
        r'^'
        r'(?:sub-(?P<sub>[a-zA-Z0-9]+)_)?'
        r'(?:ses-(?P<ses>[a-zA-Z0-9]+)_)?'
        r'(?:task-(?P<task>[a-zA-Z0-9]+)_)?'
        r'(?:tracksys-(?P<tracksys>[a-zA-Z0-9]+)_)?'
        r'(?:acq-(?P<acq>[a-zA-Z0-9]+)_)?'
        r'(?:run-(?P<run>[a-zA-Z0-9]+)_)?'
        r'(?:recording-(?P<recording>[a-zA-Z0-9]+)_)?'
        r'(?P<suffix>[a-zA-Z0-9]+)'
        r'(?P<ext>\.[a-zA-Z0-9]+)$'
    )
    match = pattern.match(stem)
    if not match:
        raise ValueError(f"Filename stem '{stem}' does not match expected BIDS pattern")

    entities = {k: v for k, v in match.groupdict().items() if v is not None}
    return entities



def get_hdf5_keys(entities: dict, run_counter: dict) -> tuple[str, str, str]:
    """
    Derive HDF5 group keys from BIDS filename entities. This ensures a consistent mapping from BIDS files to HDF5 groups,
    even when some BIDS tasks or runs are missing from some modalities.

    Returns (subject_key, session_key, run_key)
    """
    
    subject_key = f"sub-{entities['sub']}"

    # Session — use ses entity if present, otherwise synthesise session-01
    session_key = f"ses-{entities['ses']}" if 'ses' in entities else 'ses-01'

    # Each combination of subject/session/task/run should map to a unique run key.
    # Instead of using bids task/run separately, we only use runs
    run_identity = (
        subject_key,
        session_key,
        entities.get('task', ''),
        entities.get('run',  ''),
    )

    if run_identity not in run_counter:
        scope = (subject_key, session_key)
        run_counter[scope] = run_counter.get(scope, 0) + 1
        run_counter[run_identity] = run_counter[scope]

    run_key = f"run-{run_counter[run_identity]:02d}"
    return subject_key, session_key, run_key


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(subject_ids: list[str],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.1,
                  seed: int = 42) -> dict[str, str]:
    """
    Assign inter-subject splits deterministically given a seed.
    Returns dict mapping subject_id → 'train'|'val'|'test'
    """
    rng = np.random.default_rng(seed)
    ids = sorted(subject_ids)   # sort for determinism
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, round(n * train_ratio))
    n_val   = max(1, round(n * val_ratio))

    splits = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            splits[sid] = 'train'
        elif i < n_train + n_val:
            splits[sid] = 'val'
        else:
            splits[sid] = 'test'
    return splits


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@dataclass
class SegmentationConfig:
    """
    Configuration for segmenting continuous runs into gesture-level runs.

    Args:
        enabled:          Whether to segment at all
        filter_col:       Column to filter events on e.g. 'trial_type'
        filter_val:       Value to keep e.g. 'action'
        segment_col:      Column that defines segment boundaries e.g. 'gesture_id'
        exclude_vals:     Values of segment_col to skip e.g. {0} to skip rest
        attr_cols:        Event columns to promote to run.attrs
        min_samples:      Minimum segment length — shorter segments are discarded
        keep_source_run:  Whether to keep the original unsegmented run group
    """
    enabled:         bool       = False
    filter_col:      str        = 'trial_type'
    filter_val:      str        = 'action'
    segment_col:     str        = 'gesture_id'
    exclude_vals:    set        = field(default_factory=lambda: {0})
    attr_cols:       set[str]   = field(default_factory=lambda: {'gesture_id', 'repetition', 'global_gesture_id', 'exercise_id'})
    min_duration:     int        = 0.5
    keep_source_run: bool       = False


def segment_run(source_run: h5py.Group,
                sess_group: h5py.Group,
                events_df: pd.DataFrame,
                cfg: SegmentationConfig,
                sample_rate: int):
    """
    Segment a continuous run group into per-gesture run groups based on events.tsv.

    For each qualifying event:
        1. Slice all modality datasets to [sample : sample + sample_duration]
        2. Create a new run group with a unique key
        3. Copy modality datasets and run attrs
        4. Write event columns as run.attrs

    Args:
        source_run:  The continuous HDF5 run group to segment
        sess_group:  Parent session group — new runs are created here
        events_df:   Parsed events.tsv DataFrame
        cfg:         Segmentation configuration
        sample_rate: Sampling frequency in Hz
    """
    if events_df is None or events_df.empty:
        logging.warning(f"No events for {source_run.name} — skipping segmentation")
        return

    # ------------------------------------------------------------------
    # Filter events to the desired trial type
    # ------------------------------------------------------------------
    if cfg.filter_col and cfg.filter_col in events_df.columns:
        filtered = events_df[events_df[cfg.filter_col] == cfg.filter_val]
    else:
        filtered = events_df

    # Exclude unwanted segment values (e.g. rest periods)
    if cfg.segment_col in filtered.columns:
        filtered = filtered[~filtered[cfg.segment_col].isin(cfg.exclude_vals)]

    if filtered.empty:
        logging.warning(f"No qualifying events in {source_run.name} after filtering — skipping")
        return

    # ------------------------------------------------------------------
    # Determine which sample index columns to use
    # ------------------------------------------------------------------
    use_sample_cols = ('sample' in filtered.columns and 
                       'sample_duration' in filtered.columns)

    # ------------------------------------------------------------------
    # Get existing run keys to determine next available run number
    # ------------------------------------------------------------------
    existing_run_nums = []
    for k in sess_group.keys():
        if k.startswith('run-'):
            try:
                existing_run_nums.append(int(k.split('-')[1]))
            except ValueError:
                pass
    next_run_num = max(existing_run_nums, default=0) + 1

    # ------------------------------------------------------------------
    # Identify modality datasets in source run (exclude label timeseries)
    # ------------------------------------------------------------------
    label_datasets = {'gesture_id', 'global_gesture_id', 'repetition'}
    modality_keys = [k for k in source_run.keys() 
                     if isinstance(source_run[k], h5py.Dataset) 
                     and k not in label_datasets]

    # ------------------------------------------------------------------
    # Create one new run group per qualifying event
    # ------------------------------------------------------------------
    for _, event_row in filtered.iterrows():
        onset = float(event_row['onset']) 
        duration   = float(event_row['duration'])

        if duration < cfg.min_duration:
            logging.debug(f"  Skipping short segment: {duration} s < {cfg.min_duration} s")
            continue

        # ------------------------------------------------------------------
        # Create new run group
        # ------------------------------------------------------------------
        new_run_key = f"run-{next_run_num:03d}"
        next_run_num += 1

        new_run = sess_group.create_group(new_run_key)

        # Copy split attrs from source run
        for attr_key in ('inter_subject_split', 'inter_session_split',
                         'intra_session_split', 'task_name', 'side'):
            if attr_key in source_run.attrs:
                new_run.attrs[attr_key] = source_run.attrs[attr_key]

        new_run.attrs['origin_path']   = source_run.attrs.get('origin_path', 'n/a')
        new_run.attrs['source_run']    = source_run.name   # provenance — which run this came from

        # ------------------------------------------------------------------
        # Write event columns as run attrs
        # ------------------------------------------------------------------
        for col in cfg.attr_cols:
            if col in event_row.index:
                val = event_row[col]
                if pd.isna(val):
                    continue
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                new_run.attrs[col] = val

        # ------------------------------------------------------------------
        # Slice and copy each modality dataset
        # ------------------------------------------------------------------
        for mod_key in modality_keys:
            src_ds = source_run[mod_key]
            mod_sample_rate = src_ds.attrs.get('sample_rate')
            if mod_sample_rate is None:
                logging.warning(f"  No sample_rate for {src_ds.name} — cannot scale event times, skipping modality")
                continue
                
            start = int(onset * mod_sample_rate)
            end  = int((onset + duration) * mod_sample_rate)
            end = min(end, src_ds.shape[0])  # ensure we don't go past the end of the recording

            sliced  = src_ds[start:end].astype(np.float32)

            new_ds = new_run.create_dataset(mod_key, data=sliced, compression='lzf')

            # Copy all modality attrs
            for attr_key, attr_val in src_ds.attrs.items():
                new_ds.attrs[attr_key] = attr_val

            # Update n_samples to reflect the sliced length
            new_ds.attrs['n_samples']          = sliced.shape[0]
            new_ds.attrs['recording_duration'] = sliced.shape[0] / mod_sample_rate

        logging.debug(f"  Created {new_run_key}: "
                      f"{cfg.segment_col}={event_row.get(cfg.segment_col, '?')} "
                      f"samples={end-start} duration={duration:.2f} s")

    # ------------------------------------------------------------------
    # Optionally remove the source run
    # ------------------------------------------------------------------
    if not cfg.keep_source_run:
        source_run_name = source_run.name.split('/')[-1]
        del sess_group[source_run_name]
        logging.debug(f"  Deleted source run: {source_run_name}")


def get_inheritance_fpath(dir: Path, data_entities: dict, suffix: str = 'channels', ext: str = '.tsv') -> Path | None:
    """
    Find the most specific file in dir matching the given entities and regex.

    Args:
        dir: Directory to search
        data_entities: Dict of BIDS entities to match, parsed from data file name e.g. {'sub': '01', 'ses': '01', 'task': 'ExerciseB'}
        regex: Compiled regex pattern to extract entities from candidate filenames
        suffix: Expected suffix of the file (e.g. 'channels.tsv')
        ext: Expected extension of the file (e.g. '.tsv')
    """
    candidates = list(dir.glob(f'*{suffix}{ext}'))

    best_path  = None
    best_score = -1

    for fpath in candidates:
        file_entities = parse_bids_filename(fpath.name)

        score = 0
        conflict = False

        for key, val in file_entities.items():
            if key in ['suffix', 'ext']:
                continue

            if key in data_entities:
                if data_entities[key] == val:
                    score += 1
                else:
                    conflict = True
                    break

        if conflict:
            continue

        if score > best_score:
            best_score = score
            best_path  = fpath

    return best_path


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_bids_to_hdf5(cfg):
    """
    Convert a BIDS dataset to the Unified Biosignal HDF5 Format (v1.0).

    Args:
        bids_dir:     Root of the BIDS dataset
        output_path:  Output .hdf5 file path
        dataset_name: Name for the root HDF5 group e.g. "ninapro_db5"
        train_ratio:  Fraction of subjects for training
        val_ratio:    Fraction of subjects for validation
        seed:         Random seed for split assignment
    """
    bids_dir = cfg.bids_dir
    output_path = cfg.output_path
    dataset_name = cfg.dataset_name
    train_ratio = cfg.train_ratio
    val_ratio = cfg.val_ratio
    seed = cfg.seed
    primary_key = cfg.primary_key
    timeseries_events_cols = cfg.get('timeseries_events_cols')
    preferred_trial_type = cfg.get('preferred_trial_type')

    segment_on_events = cfg.get('event_segmenation') is not None and cfg.event_segmenation.enabled

    bids_dir    = Path(bids_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset-level metadata
    # ------------------------------------------------------------------
    participants = load_participants(bids_dir)

    # Assign inter-subject splits
    subject_ids = list(participants.keys()) or \
                   [p.name for p in sorted(bids_dir.glob('sub-*')) if p.is_dir()]
    split_map = assign_splits(subject_ids, train_ratio, val_ratio, seed)

    logging.info(f"Found {len(subject_ids)} subjects. Splits: "
                 f"{sum(v=='train' for v in split_map.values())} train / "
                 f"{sum(v=='val'   for v in split_map.values())} val / "
                 f"{sum(v=='test'  for v in split_map.values())} test")

    # ------------------------------------------------------------------
    # 2. Discover all data files
    # ------------------------------------------------------------------

    # FOR  TESTING AGAINST NINAPRO
    # TASK_SORT_ORDER = {
    #     'ExerciseB': 1,   # E1 in filename
    #     'ExerciseC': 2,   # E2 in filename
    #     'ExerciseA': 3,   # E3 in filename
    # }

    # all_files = sorted(
    #     bids_dir.rglob('sub-*.edf'),
    #     key=lambda p: (
    #         parse_bids_filename(p.stem).get('sub', ''),
    #         parse_bids_filename(p.stem).get('ses', ''),
    #         TASK_SORT_ORDER.get(parse_bids_filename(p.stem).get('task', ''), 99),
    #     )
    # ) + \
    #             sorted(bids_dir.rglob('sub-*_motion.tsv')) + \
    #             sorted(bids_dir.rglob('sub-*_motion.tsv.gz'))

    # all_files = []
    # for sub_dir in sorted(bids_dir.glob('sub-*')):
    #     if not sub_dir.is_dir():
    #         continue
    #     all_files += sorted(sub_dir.rglob('*.edf'))
    #     all_files += sorted(sub_dir.rglob('*_motion.tsv'))
    #     all_files += sorted(sub_dir.rglob('*_motion.tsv.gz'))


    all_files = (
        sorted(bids_dir.glob('sub-*/**/sub-*.edf')) +
        sorted(bids_dir.glob('sub-*/**/sub-*_motion.tsv')) +
        sorted(bids_dir.glob('sub-*/**/sub-*_motion.tsv.gz'))
    )

    if not all_files:
        raise FileNotFoundError(f"No EDF or motion TSV files found under {bids_dir}")

    logging.info(f"Found {len(all_files)} data files to process.")

    # ------------------------------------------------------------------
    # 3. Write HDF5
    # ------------------------------------------------------------------
    with h5py.File(output_path, 'w') as hf:
        root = hf.create_group(dataset_name)
        root.attrs['bids_dir'] = str(bids_dir)

        run_counter = {}  # track run counts per subject/session for unique run keys

        for fpath in tqdm(all_files, desc=f"Converting {dataset_name}"):
            entities = parse_bids_filename(fpath.name)
            suffix   = entities.get('suffix', '')

            if suffix not in HANDLED_SUFFIXES:
                continue

            subject_key, session_key, run_key = get_hdf5_keys(entities, run_counter)

            # --------------------------------------------------------------
            # Subject group
            # --------------------------------------------------------------
            if subject_key not in root:
                subj_group = root.create_group(subject_key)
                participant_row = participants.get(subject_key, {})
                inter_subject_split = split_map.get(subject_key, 'n/a')
                write_subject_attrs(subj_group, participant_row, inter_subject_split)
                subj_group.attrs['origin_path'] = str(fpath.parent)
            else:
                subj_group = root[subject_key]

            # --------------------------------------------------------------
            # Session group
            # --------------------------------------------------------------
            if session_key not in subj_group:
                sess_group = subj_group.create_group(session_key)
                sess_group.attrs['session_id'] = session_key
                sess_group.attrs['inter_session_split'] = 'n/a'     # requires dataset-specific knowledge
                sess_group.attrs['origin_path'] = str(fpath.parent)
            else:
                sess_group = subj_group[session_key]

            # --------------------------------------------------------------
            # Run group
            # --------------------------------------------------------------
            if run_key not in sess_group:
                run_group = sess_group.create_group(run_key)
                run_group.attrs['run_id'] = run_key
                run_group.attrs['intra_session_split'] = 'n/a'      # not implemented, requires dataset-specific knowledge
                run_group.attrs['origin_path'] = str(fpath)

                # Write task name from entities
                if 'task' in entities:
                    run_group.attrs['task_name'] = entities['task']
            else:
                run_group = sess_group[run_key]

            # --------------------------------------------------------------
            # Resolve sidecar JSON (with BIDS inheritance)
            # --------------------------------------------------------------
            sidecar = resolve_sidecar(fpath.parent, entities)

            # Write sidecar attrs to run group
            write_sidecar_attrs(run_group, sidecar)

            sample_rate = int(sidecar.get('SamplingFrequency', 
                              run_group.attrs.get('sample_rate', 0)))
            
            # --------------------------------------------------------------
            # Load channels.tsv
            # --------------------------------------------------------------
            # if channels.tsv is present, it will be used to route channels into separate modality groups (e.g. emg, ecg, accel).
            channels_tsv_path = get_inheritance_fpath(fpath.parent, entities, suffix='channels', ext='.tsv')
            channels_df = load_channels_tsv(channels_tsv_path)

            # --------------------------------------------------------------
            # Read signal data
            # --------------------------------------------------------------
            if fpath.suffix == '.edf':
                modality_arrays = read_edf(fpath, channels_df)   
            elif suffix == 'motion':
                modality_arrays = read_motion_tsv(fpath, channels_df)
            else:
                logging.warning(f"Unhandled file type: {fpath} — skipping")
                continue

            # --------------------------------------------------------------
            # Write modalities
            # --------------------------------------------------------------
            for group_name, data in modality_arrays.items():
                if group_name in run_group:
                    logging.warning(f"  Modality '{group_name}' already exists in "
                                    f"{run_group.name} — skipping duplicate")
                    continue
                write_modality(run_group, group_name, data, sidecar, str(fpath))

            # --------------------------------------------------------------
            # Load events.tsv
            # --------------------------------------------------------------
            events_tsv_path = None
            events_dir = fpath.parent
            while events_tsv_path is None:
                events_tsv_path = get_inheritance_fpath(events_dir, entities, suffix='events', ext='.tsv')
                # Check if we are in the root of the dataset by looking for the presence of participants.tsv
                if (events_dir / 'participants.tsv').exists():
                    break
                events_dir = events_dir.parent
            
            events_df = load_events_tsv(events_tsv_path)

            # --------------------------------------------------------------
            # Parse events into run attrs
            # --------------------------------------------------------------
            if events_df is not None:
                if primary_key in run_group:
                    n_samples = run_group[primary_key].shape[0]
                    if sample_rate == 0:
                        sample_rate = run_group[primary_key].attrs.get('sample_rate', 1000)
                        logging.warning(f"Sample rate not found in sidecar or run attrs for {run_group.name} — using {sample_rate} from primary modality {primary_key}")
                        print(sample_rate)
                    
                    parse_events(
                        events_df=events_df,
                        run_group=run_group,
                        n_samples=n_samples,
                        sample_rate=sample_rate,
                        timeseries_cols=set(timeseries_events_cols),   
                        preferred_trial_type=preferred_trial_type,  
                    )
                else:
                    logging.warning(f"Cannot parse events.tsv for {run_group.name} — primary modality '{primary_key}' not found, so sample count is unknown.")

        logging.info(f"Conversion complete → {output_path}")
        logging.info(f"Root groups: {list(root.keys())[:5]}...")


    if segment_on_events:
        seg_cfg = SegmentationConfig(**cfg.event_segmenation)
        logging.info("Running segmentation pass...")

        with h5py.File(output_path, 'a') as hf:   # open in append mode
            root = hf[dataset_name]

            for subj_key in tqdm(root.keys(), desc="Segmenting"):
                subj_group = root[subj_key]
                for sess_key in subj_group.keys():
                    sess_group = subj_group[sess_key]
                    for run_key in list(sess_group.keys()):   # list() — keys may change during loop
                        run_group = sess_group[run_key]

                        # Reconstruct events path from origin_path attr
                        origin = Path(run_group.attrs.get('origin_path', ''))
                        base = origin.stem   # e.g. "sub-01_task-ExerciseB_emg"
                        # Strip modality suffix to get base entity string
                        base_entities = '_'.join(
                            p for p in base.split('_') 
                            if not any(p == s for s in HANDLED_SUFFIXES)
                        )
                        events_path = origin.parent / f"{base_entities}_events.tsv"
                        events_df   = load_events_tsv(events_path)

                        sample_rate = run_group.attrs.get('sample_rate', 0)
                        if sample_rate == 0 and primary_key in run_group:
                            sample_rate = int()

                        segment_run(
                            source_run  = run_group,
                            sess_group  = sess_group,
                            events_df   = events_df,
                            cfg         = seg_cfg,
                            sample_rate = sample_rate,
                        )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BIDS dataset to HDF5')
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file with conversion parameters")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    convert_bids_to_hdf5(cfg)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Convert BIDS dataset to HDF5')
#     parser.add_argument('--bids_dir',     type=str, required=True)
#     parser.add_argument('--output_path',  type=str, required=True)
#     parser.add_argument('--dataset_name', type=str, required=True)
#     parser.add_argument('--train_ratio',  type=float, default=0.7)
#     parser.add_argument('--val_ratio',    type=float, default=0.1)
#     parser.add_argument('--seed',         type=int,   default=42)
#     parser.add_argument('--log_level',    type=str,   default='INFO')
#     args = parser.parse_args()

#     logging.basicConfig(level=getattr(logging, args.log_level.upper()),
#                         format='%(asctime)s %(levelname)s %(message)s')

#     convert_bids_to_hdf5(
#         bids_dir     = args.bids_dir,
#         output_path  = args.output_path,
#         dataset_name = args.dataset_name,
#         train_ratio  = args.train_ratio,
#         val_ratio    = args.val_ratio,
#         seed         = args.seed,
#     )