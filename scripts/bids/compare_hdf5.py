""" Script compare_hdf5.py

Compares two HDF5 files produced by different conversion pipelines to verify
they contain equivalent data and metadata. Designed to validate that the
BIDS → HDF5 conversion produces the same output as the direct → HDF5 conversion.

Usage:
    python -m scripts.bids.compare_hdf5 \
        --reference data/ninapro_db5/ninapro_db5.hdf5 \
        --candidate data/ninapro_db5_bids/ninapro_db5.hdf5 \
        --dataset_name ninapro_db5 \
        --rtol 1e-4 \
        --atol 1e-6

Date:
    11/05/2026
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Mismatch:
    path:    str
    kind:    str        # "shape", "dtype", "values", "attr", "missing", "extra"
    detail:  str


@dataclass
class ComparisonReport:
    mismatches:     list[Mismatch] = field(default_factory=list)
    warnings:       list[str]      = field(default_factory=list)
    datasets_checked: int = 0
    attrs_checked:    int = 0

    def add_mismatch(self, path: str, kind: str, detail: str):
        self.mismatches.append(Mismatch(path, kind, detail))

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.mismatches) == 0

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "HDF5 COMPARISON REPORT",
            "=" * 70,
            f"  Datasets checked : {self.datasets_checked}",
            f"  Attrs checked    : {self.attrs_checked}",
            f"  Mismatches       : {len(self.mismatches)}",
            f"  Warnings         : {len(self.warnings)}",
            f"  Result           : {'✅ PASS' if self.passed else '❌ FAIL'}",
        ]

        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")

        if self.mismatches:
            lines.append("\nMISMATCHES:")
            # Group by kind for readability
            by_kind: dict[str, list[Mismatch]] = {}
            for m in self.mismatches:
                by_kind.setdefault(m.kind, []).append(m)

            for kind, items in by_kind.items():
                lines.append(f"\n  [{kind.upper()}] ({len(items)} issues)")
                for m in items[:20]:    # cap at 20 per kind to avoid flooding
                    lines.append(f"    {m.path}")
                    lines.append(f"      → {m.detail}")
                if len(items) > 20:
                    lines.append(f"    ... and {len(items) - 20} more")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Attribute comparison
# ---------------------------------------------------------------------------

# Attrs that are expected to differ between pipelines — don't flag these
IGNORED_ATTR_KEYS = {
    'origin_path',      # will differ — different source paths
    'bids_dir',         # only in BIDS-derived file
    'sensor_type',      # may differ due to different sidecar keys (Manufacturer vs ManufacturersModelName)
    'n_samples',       # may differ due to EDF padding differences
    'recording_duration', # may differ due to EDF padding differences
}

# Attrs that may be present in one but not the other — warn but don't fail
OPTIONAL_ATTR_KEYS = {
    'weight',
    'height',
    'manufacturer',
    'powerline_frequency',
    'task_name',
}


def compare_attrs(ref_attrs, cand_attrs, path: str, report: ComparisonReport):
    """
    Compare HDF5 attribute dicts at a given path.
    Ignores keys in IGNORED_ATTR_KEYS.
    Warns on keys in OPTIONAL_ATTR_KEYS.
    """
    ref_keys  = {k for k in ref_attrs.keys()  if k not in IGNORED_ATTR_KEYS}
    cand_keys = {k for k in cand_attrs.keys() if k not in IGNORED_ATTR_KEYS}

    # Keys only in reference
    for k in ref_keys - cand_keys:
        if k in OPTIONAL_ATTR_KEYS:
            report.add_warning(f"{path}.attrs['{k}'] present in reference but missing in candidate")
        else:
            report.add_mismatch(path, 'attr',
                f"Key '{k}' present in reference but missing in candidate")

    # Keys only in candidate
    for k in cand_keys - ref_keys:
        if k in OPTIONAL_ATTR_KEYS:
            report.add_warning(f"{path}.attrs['{k}'] present in candidate but missing in reference")
        else:
            report.add_mismatch(path, 'attr',
                f"Key '{k}' present in candidate but missing in reference")

    # Keys in both — compare values
    for k in ref_keys & cand_keys:
        report.attrs_checked += 1
        ref_val  = ref_attrs[k]
        cand_val = cand_attrs[k]

        # Normalise numpy scalars to Python types for comparison
        if isinstance(ref_val,  np.generic): ref_val  = ref_val.item()
        if isinstance(cand_val, np.generic): cand_val = cand_val.item()

        # String comparison — case-insensitive, strip whitespace
        if isinstance(ref_val, str) and isinstance(cand_val, str):
            if ref_val.strip().lower() != cand_val.strip().lower():
                report.add_mismatch(path, 'attr',
                    f"'{k}': reference='{ref_val}' candidate='{cand_val}'")

        # Numeric comparison
        elif isinstance(ref_val, (int, float)) and isinstance(cand_val, (int, float)):
            if not np.isclose(ref_val, cand_val, rtol=1e-4, atol=1e-6):
                report.add_mismatch(path, 'attr',
                    f"'{k}': reference={ref_val} candidate={cand_val}")

        # Fallback — direct equality
        else:
            if ref_val != cand_val:
                report.add_mismatch(path, 'attr',
                    f"'{k}': reference={ref_val!r} candidate={cand_val!r}")


# ---------------------------------------------------------------------------
# Dataset (signal array) comparison
# ---------------------------------------------------------------------------

def compare_dataset(ref_ds: h5py.Dataset, cand_ds: h5py.Dataset,
                    path: str, report: ComparisonReport,
                    rtol: float, atol: float):
    """Compare shape, dtype, and values of two HDF5 datasets."""
    report.datasets_checked += 1

    # Shape
    if ref_ds.shape[0] != cand_ds.shape[0]:
        diff = abs(ref_ds.shape[0] - cand_ds.shape[0])
        sample_rate = ref_ds.attrs.get('sample_rate', 200)
        
        if diff < sample_rate:
            # Less than 1 second difference — likely EDF record padding, warn not fail
            report.add_warning(
                f"{path}: shape differs by {diff} samples "
                f"(<1s at {sample_rate}Hz) — likely EDF record boundary padding"
            )
        else:
            report.add_mismatch(path, 'shape',
                f"reference={ref_ds.shape} candidate={cand_ds.shape}")
            return
    
    # Dtype — warn only, different pipelines may use slightly different dtypes
    if ref_ds.dtype != cand_ds.dtype:
        report.add_warning(f"{path}: dtype mismatch — "
                           f"reference={ref_ds.dtype} candidate={cand_ds.dtype}")

    # Values
    ref_data  = ref_ds[()].astype(np.float64)
    cand_data = cand_ds[()].astype(np.float64)
    # Truncate to the shorter length if shapes differ, to avoid spurious value mismatches due to padding of edf
    min_length = min(ref_data.shape[0], cand_data.shape[0])
    ref_data  = ref_data[:min_length]
    cand_data = cand_data[:min_length]

    if not np.allclose(ref_data, cand_data, rtol=rtol, atol=atol, equal_nan=True):
        max_abs_diff = np.abs(ref_data - cand_data).max()
        max_rel_diff = (np.abs(ref_data - cand_data) / 
                        (np.abs(ref_data) + 1e-10)).max()
        
        # Find worst offending location
        worst_idx = np.unravel_index(
            np.abs(ref_data - cand_data).argmax(), ref_data.shape
        )
        report.add_mismatch(path, 'values',
            f"max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}, "
            f"worst_at={worst_idx} "
            f"(ref={ref_data[worst_idx]:.6f}, cand={cand_data[worst_idx]:.6f})")

    # Compare modality-level attrs
    compare_attrs(ref_ds.attrs, cand_ds.attrs, path, report)


# ---------------------------------------------------------------------------
# Recursive group traversal
# ---------------------------------------------------------------------------

def compare_groups(ref_group: h5py.Group, cand_group: h5py.Group,
                   path: str, report: ComparisonReport,
                   rtol: float, atol: float):
    """
    Recursively compare two HDF5 groups — their attrs, child groups,
    and datasets.
    """
    # Compare attrs at this level
    compare_attrs(ref_group.attrs, cand_group.attrs, path, report)

    ref_keys  = set(ref_group.keys())
    cand_keys = set(cand_group.keys())

    # Keys only in reference
    for k in sorted(ref_keys - cand_keys):
        report.add_mismatch(f"{path}/{k}", 'missing',
            f"Present in reference but missing in candidate")

    # Keys only in candidate
    for k in sorted(cand_keys - ref_keys):
        report.add_mismatch(f"{path}/{k}", 'extra',
            f"Present in candidate but missing in reference")

    # Keys in both
    for k in sorted(ref_keys & cand_keys):
        child_path = f"{path}/{k}"
        ref_child  = ref_group[k]
        cand_child = cand_group[k]

        # Type mismatch — one is a group, the other a dataset
        if type(ref_child) != type(cand_child):
            report.add_mismatch(child_path, 'missing',
                f"Type mismatch: reference={type(ref_child).__name__} "
                f"candidate={type(cand_child).__name__}")
            continue

        if isinstance(ref_child, h5py.Group):
            compare_groups(ref_child, cand_child, child_path, report, rtol, atol)

        elif isinstance(ref_child, h5py.Dataset):
            compare_dataset(ref_child, cand_child, child_path, report, rtol, atol)


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------

def compare_hdf5(reference_path: Path,
                 candidate_path: Path,
                 dataset_name: str,
                 rtol: float = 1e-4,
                 atol: float = 1e-6) -> ComparisonReport:
    """
    Compare two HDF5 files under a shared root group name.

    Args:
        reference_path: Path to the ground-truth HDF5 (direct conversion)
        candidate_path: Path to the HDF5 to validate (BIDS → HDF5)
        dataset_name:   Root group name e.g. "ninapro_db5"
        rtol:           Relative tolerance for numeric comparisons
        atol:           Absolute tolerance for numeric comparisons

    Returns:
        ComparisonReport with all mismatches and warnings
    """
    report = ComparisonReport()

    with h5py.File(reference_path, 'r') as ref_f, \
         h5py.File(candidate_path, 'r') as cand_f:

        # Check root group exists in both
        if dataset_name not in ref_f:
            raise KeyError(f"Root group '{dataset_name}' not found in reference file")
        if dataset_name not in cand_f:
            raise KeyError(f"Root group '{dataset_name}' not found in candidate file")

        ref_root  = ref_f[dataset_name]
        cand_root = cand_f[dataset_name]

        logging.info(f"Reference : {reference_path} "
                     f"({len(ref_root)} subjects)")
        logging.info(f"Candidate : {candidate_path} "
                     f"({len(cand_root)} subjects)")

        compare_groups(ref_root, cand_root, dataset_name, report, rtol, atol)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare two HDF5 biosignal files for equivalence'
    )
    parser.add_argument('--reference',    type=str, required=True,
                        help='Ground-truth HDF5 (direct conversion)')
    parser.add_argument('--candidate',    type=str, required=True,
                        help='HDF5 to validate (BIDS → HDF5)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Root group name e.g. ninapro_db5')
    parser.add_argument('--rtol',         type=float, default=1e-4)
    parser.add_argument('--atol',         type=float, default=1e-6)
    parser.add_argument('--log_level',    type=str,   default='INFO')
    parser.add_argument('--output',       type=str,   default=None,
                        help='Optional path to save report as .txt')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s %(levelname)s %(message)s'
    )

    report = compare_hdf5(
        reference_path = Path(args.reference),
        candidate_path = Path(args.candidate),
        dataset_name   = args.dataset_name,
        rtol           = args.rtol,
        atol           = args.atol,
    )

    print(report.summary())

    if args.output:
        Path(args.output).write_text(report.summary())
        logging.info(f"Report saved to {args.output}")

    # Exit with non-zero code if comparison failed — useful for CI
    exit(0 if report.passed else 1)