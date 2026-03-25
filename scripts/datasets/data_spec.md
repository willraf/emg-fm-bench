# Technical Specification: Unified Biosignal HDF5 Format (v1.0)

This document defines the standardized HDF5 structure for all biosignal datasets (EPN612, NinaPro, emg2pose) used in this project. Adherence to this schema ensures compatibility with the `HDF5WindowedDataset` and allows for multi-dataset training without code modification.

---

## 1. Directory & Internal Hierarchy

Datasets are stored as single HDF5 files per dataset. The hierarchy follows a subject and session split, treating sessions/trials as discrete groups.

```text
/ [dataset_name] (Root)
└── [subject_id] (Group)             # e.g., "S1", "u01", "d3870"
    ├── .attrs (Subject Metadata)    # Demographic and physical data, user split
    └── [session_id] (Group)         # e.g., "E1_RR1_RG1", "session1"
        ├── .attrs (Session Metadata)# Trial-specific data (gesture, split)
        ├── emg (Dataset)            # shape: (T, C), dtype: float32
            ├── .attrs               # Sample rate for the modality
        ├── imu (Dataset)            # shape: (T, D), dtype: float32 or Compound
        └── kinematics (Dataset)     # shape: (T, K), dtype: float32 (Glove/Joints)

```

---

## 2. Standardized Attributes (Metadata)

To enable automated manifest generation, the following keys are **Reserved**. During conversion, native field names must be mapped to these keys.

### 2.1 Subject Level (`[subject_id].attrs`)

| Key             | Type   | Units/Values                  | Description                   |
| --------------- | ------ | ----------------------------- | ----------------------------- |
| `subject_id`    | String | -                             | Unique subject identifier.    |
| `gender`        | String | `m`, `f`, `o`, `u`            | Male, Female, Other, Unknown. |
| `age`           | Int    | Years                         | Subject age.                  |
| `handedness`    | String | `r`, `l`, `a`                 | Right, Left, Ambidextrous.    |
| `arm_perimeter` | Float  | cm                            | Forearm circumference.        |
| `ethnicity`     | String | -                             | Subject ethnic group/origin.  |
| `subject_split` | String | `train`, `val`, `test`, `n/a` | Inter subject level split     |

### 2.2 Session Level (`[session_id].attrs`)

| Key                   | Type   | Units/Values                  | Description                                                                                                                                    |
| --------------------- | ------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `session_id`          | String | -                             | Unique trial, repetition, or recording ID.                                                                                                     |
| `gesture_id`          | Int    | -                             | **Global ID** (Mapped across Exercises 1, 2, 3 for NinaPro). Gestures are at indexed within the dataset rather than globally between datasets. |
| `sample_rate`         | Int    | Hz                            | Sampling frequency in Hz of the primary datatype (standardized to EMG).                                                                        |
| `exercise_id`         | Int    |                               | NinaPro-specific exercise category (1, 2, or 3).                                                                                               |
| `intra_session_split` | String | `train`, `val`, `test`, `n/a` | Training and testing from the same session                                                                                                     |
| `inter_session_split` | String | `train`, `val`, `test`, `n/a` | Training and testing from the same user, different session                                                                                     |
| `subject_split`       | String | `train`, `val`, `test`, `n/a` | Training and testing from the different users (same as `[subject_id].attrs.split`)                                                             |

---

### 2.3 Modality Level (`[modality_name].attrs`)
| Key           | Type   | Description                             |
| ------------- | ------ | --------------------------------------- |
| `sample_rate` | Int    | Sample rate used for this modality (Hz) |
| `sensor_type` | String | Name of the sensor used for recordings  |
| `units`       | String | Units if knows, or "normalised"         |

## 3. Data Modality Standards

### 3.1 EMG (`/emg`)

* **Storage:** Continuous 2D array of shape `(Time, Channels)`.
* **Precision:** `float32`.
* **Normalization:** All signals must be scaled to the range `[-1.0, 1.0]`.
* *EPN/Myo conversion: divide raw 8-bit values by 128.0.*


* **Attributes:**
* `units`: `"normalized"`.
* `sample_rate`: in Hertz
* `sensor_type`


### 3.2 Accelerometer (`/accelerometer`)
* **Storage:** 2D array of shape `(Time, Axis)` where axis will usually be `[x, y, z]`
* **Precision:** `float32`

### 3.3 Gyroscope (`/gyroscope`)
* **Storage:** 2D array of shape `(Time, Axis)` where axis will usually be `[x, y, z]`
* **Precision:** `float32`

### 3.4 Quaternion (`/quaternion`)
* **Storage:** 2D array of shape `(Time, Axis)` where axis will usually be `[w, x, y, z]`
* **Precision:** `float32`

### 3.5 Kinematics (`/kinematics`)

* **Source:** Cyberglove (NinaPro) or Joint Angles (emg2pose).
* **Storage:** 2D array of shape `(Time, DoF)`.

---

## 4. Dataset Rosetta Stone (Mapping Table)

| Unified Key     | EPN612 Native              | NinaPro DB5 Native                                           | emg2pose Native |
| --------------- | -------------------------- | ------------------------------------------------------------ | --------------- |
| `sample_rate`   | `samplingFrequencyInHertz` | `frequency`                                                  | `sample_rate`   |
| `arm_perimeter` | `armPerimeterInCm`         | `circumference_cm`                                           | N/A             |
| `handedness`    | `handedness`               | `laterality`                                                 | N/A             |
| `gesture_id`    | `session` (mapped)         | `restimulus` + exercise based offset (for global gesture id) | N/A             |
|                 |                            |                                                              |                 |

---

## 5. The Global Manifest

__TODO__ (not yet implemented)
A `manifest.csv` is generated by crawling the standardized HDF5 files. This CSV acts as the "Catalog" for the PyTorch `Dataset`.

**Required Manifest Columns:**
`dataset`, `subject_id`, `session_id`, `h5_path`, `total_samples`, `sample_rate`, `gesture_id`, `split`.

---

## 6. Usage Protocol
__TODO__ (not yet implemented)
1. **Conversion:** Run dataset-specific conversion scripts to produce the HDF5 structure above.
2. **Validation:** Run `validate_hdf5.py` to ensure all mandatory `.attrs` exist.
3. **Indexing:** Run `build_manifest.py` to update the global CSV index.
4. **Training:** Initialize `UniversalManifestDataset(manifest_path="manifest.csv")`.
