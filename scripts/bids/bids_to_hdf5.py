""" Module bids_to_hdf5.py

Universal converter from BIDS format to the Unified Biosignal HDF5 Format (v1.0).

Supports modalities: EMG, EEG, Motion (ACCEL, GYRO, ORNT/quaternion, JNTANG)
Enforces: subject --> session --> run --> modality hierarchy

Usage:
    python -m scripts.bids.bids_to_hdf5 \
        --bids_dir data/ninapro_db5_bids/ \
        --output_path data/ninapro_db5_bids/ninapro_db5.hdf5 \
        --dataset_name ninapro_db5

Authors:
    Will Raftery
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

# BIDS motion channels.tsv type → HDF5 group name
MOTION_TYPE_TO_GROUP = {
    'ACCEL':   'accel',
    'GYRO':    'gyro',
    'ORNT':    'ornt',
    'JNTANG':  'jntang',
    'MAGN':    'magn',
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
    return None


def load_channels_tsv(path: Path) -> pd.DataFrame | None:
    """Load a BIDS channels.tsv, return None if missing."""
    if path.exists():
        return pd.read_csv(path, sep='\t')
    return None


def resolve_sidecar(bids_dir: Path, subject: str, session: str | None,
                    suffix: str, entities: dict) -> dict:
    """
    Resolve BIDS sidecar JSON with inheritance.
    
    BIDS inheritance works by walking UP the directory tree from the file,
    merging JSONs at each level. More specific (closer to file) wins.
    
    At each directory level, valid sidecar filenames are built from
    optional entity combinations in this order:
        [sub]-[ses]-[task]-[acq]-[run]-[tracksys]-[suffix].json
    
    Inheritance order (least → most specific):
        root → sub/ → sub/ses/ → sub/ses/modality/ (file-level)
    """
    merged = {}

    def _load_and_merge(path: Path):
        """Load JSON at path and merge into accumulated dict. 
        More specific keys already in merged are NOT overwritten."""
        data = load_sidecar_json(path)
        if data:
            logging.debug(f"  Loaded sidecar: {path}")
        merged.update({k: v for k, v in data.items() if k not in merged})

    # ------------------------------------------------------------------
    # Build ordered list of candidate sidecar filenames at a given dir.
    # BIDS spec: entities are included only if present, in canonical order.
    # More entities = more specific = checked last = wins on conflict.
    # ------------------------------------------------------------------
    def candidate_sidecars(directory: Path, 
                           include_sub: bool = False,
                           include_ses: bool = False) -> list[Path]:
        """
        Generate candidate sidecar paths at a directory level,
        ordered from least to most specific.
        """
        # Canonical BIDS entity order for sidecar matching
        # Only include entities that are present in this file's entities dict
        sub     = entities.get('sub')
        ses     = entities.get('ses')
        task    = entities.get('task')
        acq     = entities.get('acq')
        run     = entities.get('run')
        tracksys = entities.get('tracksys')

        # Build entity combinations from least to most specific
        # Each tuple is a sequence of (key, value) pairs to include
        candidates = []

        # 1. Suffix only — least specific, applies to all files of this type
        candidates.append([])

        # 2. Task only
        if task:
            candidates.append([('task', task)])

        # 3. Acq only
        if acq:
            candidates.append([('acq', acq)])

        # 4. Task + acq
        if task and acq:
            candidates.append([('task', task), ('acq', acq)])

        # 5. Task + run
        if task and run:
            candidates.append([('task', task), ('run', run)])

        # 6. Task + tracksys (motion-specific)
        if task and tracksys:
            candidates.append([('task', task), ('tracksys', tracksys)])

        # 7. Task + acq + run
        if task and acq and run:
            candidates.append([('task', task), ('acq', acq), ('run', run)])

        # 8. Task + acq + tracksys
        if task and acq and tracksys:
            candidates.append([('task', task), ('acq', acq), ('tracksys', tracksys)])

        # 9. Full specificity — all present entities
        full = []
        if include_sub and sub: full.append(('sub', sub))
        if include_ses and ses: full.append(('ses', ses))
        if task:                full.append(('task', task))
        if acq:                 full.append(('acq', acq))
        if run:                 full.append(('run', run))
        if tracksys:            full.append(('tracksys', tracksys))
        if full not in candidates:
            candidates.append(full)

        # Build actual paths
        paths = []
        for entity_pairs in candidates:
            if entity_pairs:
                fname = '_'.join(f'{k}-{v}' for k, v in entity_pairs) + f'_{suffix}.json'
            else:
                fname = f'{suffix}.json'
            paths.append(directory / fname)

        return paths

    # ------------------------------------------------------------------
    # Walk from least to most specific, loading and merging at each level
    # ------------------------------------------------------------------

    # 1. Root level — no subject/session prefix
    for path in candidate_sidecars(bids_dir):
        _load_and_merge(path)

    # 2. Subject level — files prefixed with sub-XX
    sub_dir = bids_dir / subject
    for path in candidate_sidecars(sub_dir, include_sub=True):
        _load_and_merge(path)

    # 3. Session level — files prefixed with sub-XX_ses-XX
    if session:
        ses_dir = sub_dir / session
        for path in candidate_sidecars(ses_dir, include_sub=True, include_ses=True):
            _load_and_merge(path)

        # 4. Modality folder level — most specific
        # BIDS stores data files in modality subfolders (emg/, motion/, eeg/ etc.)
        modality_dir = ses_dir / suffix
        for path in candidate_sidecars(modality_dir, include_sub=True, include_ses=True):
            _load_and_merge(path)

    else:
        # No session — modality folder is directly under subject
        modality_dir = sub_dir / suffix
        for path in candidate_sidecars(modality_dir, include_sub=True):
            _load_and_merge(path)

    return merged


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
    Read an EDF file using MNE. Returns a dict mapping modality group name
    to (T, C) float32 array.

    If channels_tsv is provided, uses the 'type' column to route channels
    into separate arrays. Otherwise puts everything under the file suffix.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    data, _ = raw.get_data(return_times=True)   # (n_channels, n_times)
    data = data.T.astype(np.float32)            # → (T, C)
    ch_names = raw.ch_names

    if channels_tsv is None:
        # No channel type info — return everything as one array
        suffix = edf_path.stem.split('_')[-1]   # e.g. "emg" from "sub-01_task-X_emg"
        return {suffix: data}

    # Route channels by type using channels.tsv
    type_col = channels_tsv['type'].str.upper() if 'type' in channels_tsv.columns else None

    if type_col is None:
        suffix = edf_path.stem.split('_')[-1]
        return {suffix: data}

    result = {}
    for ch_idx, ch_name in enumerate(ch_names):
        # Find this channel in channels.tsv
        row = channels_tsv[channels_tsv['name'] == ch_name]
        if row.empty:
            ch_type = 'unknown'
        else:
            ch_type = row['type'].values[0].upper()

        # Map BIDS type to HDF5 group name
        group_name = MOTION_TYPE_TO_GROUP.get(ch_type, ch_type.lower())

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
        group_name = MOTION_TYPE_TO_GROUP.get(ch_type, ch_type.lower())

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

DEFAULT_TIMESERIES_COLS = {'gesture_id', 'global_gesture_id', 'repetition'}
PREFERRED_TRIAL_TYPE = 'action'   # prefer 'action' events for labels, fallback to all if not present

def reconstruct_timeseries_events(events_df: pd.DataFrame,
                                    n_samples: int,
                                    sample_rate: int,
                                    timeseries_cols: set[str] = DEFAULT_TIMESERIES_COLS,
                                    preferred_trial_type: str = PREFERRED_TRIAL_TYPE,
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
                              timeseries_cols: set[str] = DEFAULT_TIMESERIES_COLS):
    """
    Extract timeseries events from events_df and write as datasets under run_group.
    """
    if events_df is None:
        return

    timeseries_arrays = reconstruct_timeseries_events(events_df, n_samples, sample_rate, timeseries_cols)

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
    Parse a BIDS filename stem into entity key-value pairs.
    e.g. "sub-01_ses-01_task-ExerciseB_run-01_emg"
    → {'sub': '01', 'ses': '01', 'task': 'ExerciseB', 'run': '01', 'suffix': 'emg'}
    """
    parts = stem.split('_')
    entities = {}
    for part in parts:
        if '-' in part:
            k, v = part.split('-', 1)
            entities[k] = v
        else:
            entities['suffix'] = part   # last part with no '-' is the suffix
    return entities


def get_hdf5_keys(entities: dict, run_counter: dict) -> tuple[str, str, str]:
    """
    Derive HDF5 group keys from BIDS filename entities.
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
# Main conversion
# ---------------------------------------------------------------------------

def convert_bids_to_hdf5(bids_dir: Path,
                          output_path: Path,
                          dataset_name: str,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.1,
                          seed: int = 42,
                          primary_key: str = 'emg'):
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
    bids_dir    = Path(bids_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset-level metadata
    # ------------------------------------------------------------------
    participants = load_participants(bids_dir)

    # Assign inter-subject splits
    subject_ids  = list(participants.keys()) or \
                   [p.name for p in sorted(bids_dir.glob('sub-*')) if p.is_dir()]
    split_map    = assign_splits(subject_ids, train_ratio, val_ratio, seed)

    logging.info(f"Found {len(subject_ids)} subjects. Splits: "
                 f"{sum(v=='train' for v in split_map.values())} train / "
                 f"{sum(v=='val'   for v in split_map.values())} val / "
                 f"{sum(v=='test'  for v in split_map.values())} test")

    # ------------------------------------------------------------------
    # 2. Discover all data files
    # ------------------------------------------------------------------


    TASK_SORT_ORDER = {
        'ExerciseB': 1,   # E1 in filename
        'ExerciseC': 2,   # E2 in filename
        'ExerciseA': 3,   # E3 in filename
    }

    all_files = sorted(
        bids_dir.rglob('sub-*.edf'),
        key=lambda p: (
            parse_bids_filename(p.stem).get('sub', ''),
            parse_bids_filename(p.stem).get('ses', ''),
            TASK_SORT_ORDER.get(parse_bids_filename(p.stem).get('task', ''), 99),
        )
    ) + \
                sorted(bids_dir.rglob('sub-*_motion.tsv')) + \
                sorted(bids_dir.rglob('sub-*_motion.tsv.gz'))
    # # Collect all EDF and motion TSV files under sub-* directories
    # all_files = sorted(bids_dir.rglob('sub-*.edf')) + \
    #             sorted(bids_dir.rglob('sub-*_motion.tsv')) + \
    #             sorted(bids_dir.rglob('sub-*_motion.tsv.gz'))

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
            entities = parse_bids_filename(fpath.stem)
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
                run_group.attrs['intra_session_split'] = 'n/a'
                run_group.attrs['origin_path'] = str(fpath)

                # Write task name from entities
                if 'task' in entities:
                    run_group.attrs['task_name'] = entities['task']
            else:
                run_group = sess_group[run_key]

            # --------------------------------------------------------------
            # Resolve sidecar JSON (with BIDS inheritance)
            # --------------------------------------------------------------
            session_str = f"ses-{entities['ses']}" if 'ses' in entities else None
            sidecar = resolve_sidecar(bids_dir, subject_key, session_str, suffix, entities)

            # Write sidecar attrs to run group
            write_sidecar_attrs(run_group, sidecar)

            sample_rate = int(sidecar.get('SamplingFrequency', 
                              run_group.attrs.get('sample_rate', 0)))

            # --------------------------------------------------------------
            # Load channels.tsv
            # --------------------------------------------------------------
            channels_tsv_path = fpath.parent / fpath.name.replace(
                f'_{suffix}.edf', f'_{suffix}_channels.tsv'
            ).replace(
                f'_{suffix}.tsv', f'_{suffix}_channels.tsv'
            ).replace(
                f'_{suffix}.tsv.gz', f'_{suffix}_channels.tsv'
            )
            # Cleaner approach — reconstruct from stem
            base = '_'.join(
                f'{k}-{v}' for k, v in entities.items()
                if k not in ('suffix',)
            )
            channels_tsv_path = fpath.parent / f'{base}_channels.tsv'
            channels_df = load_channels_tsv(channels_tsv_path)

            # --------------------------------------------------------------
            # Load events.tsv
            # --------------------------------------------------------------
            events_tsv_path = fpath.parent / f'{base}_events.tsv'
            events_df = load_events_tsv(events_tsv_path)
            
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
            # Parse events into run attrs
            # --------------------------------------------------------------
            if events_df is not None:
                if primary_key in run_group:
                    n_samples = run_group[primary_key].shape[0]
                    if sample_rate == 0:
                        sample_rate = run_group[primary_key].attrs.get('sample_rate', 1000)
                        logging.warning(f"Sample rate not found in sidecar or run attrs for {run_group.name} — using {sample_rate} from primary modality")
                        print(sample_rate)
                    
                    parse_events(
                        events_df=events_df,
                        run_group=run_group,
                        n_samples=n_samples,
                        sample_rate=sample_rate,
                        timeseries_cols=DEFAULT_TIMESERIES_COLS,   # override per dataset if needed
                    )
                else:
                    logging.warning(f"Cannot parse events.tsv for {run_group.name} — primary modality '{primary_key}' not found, so sample count is unknown.")

        logging.info(f"Conversion complete → {output_path}")
        logging.info(f"Root groups: {list(root.keys())[:5]}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BIDS dataset to HDF5')
    parser.add_argument('--bids_dir',     type=str, required=True)
    parser.add_argument('--output_path',  type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--train_ratio',  type=float, default=0.7)
    parser.add_argument('--val_ratio',    type=float, default=0.1)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_level',    type=str,   default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s %(levelname)s %(message)s')

    convert_bids_to_hdf5(
        bids_dir     = args.bids_dir,
        output_path  = args.output_path,
        dataset_name = args.dataset_name,
        train_ratio  = args.train_ratio,
        val_ratio    = args.val_ratio,
        seed         = args.seed,
    )