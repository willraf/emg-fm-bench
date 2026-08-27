import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
TUNING_ORDER = ["Full FT", "From-Scratch FT", "LP", "From-Scratch LP"]
MODEL_ORDER  = ["LDA", "SVM", "RF", "CNN", "NeuroRVQ", "PhysioWave", "TinyMyo"]

MODEL_MAP = {
    "tinymyo":    "TinyMyo",
    "neurorvq":   "NeuroRVQ",
    "physiowave": "PhysioWave",
    "lda":        "LDA",
    "svm":        "SVM",
    "rf":         "RF",
    "cnn":        "CNN",
}

DATASET_MAP = {
    "classification": {
        "epn612":      "EPN-612",
        "ninapro_db5": "NinaPro DB5",
        "uci_emg":     "UCI-EMG",
    },
    "regression": {
        "ninapro_db5":      "NinaPro DB5",
        "neuromotor_wrist": "Neuromotor-Wrist",
        "ninapro_db2":      "NinaPro DB2",
    },
}

# ------------------------------------------------------------------ #
# Task registry
# Each metric entry defines:
#   col:             column name in results.csv (without "test_" prefix after stripping)
#   header:          LaTeX column header
#   scale:           multiply mean/std by this before formatting
#   higher_is_better: used for best-flagging direction
# ------------------------------------------------------------------ #
TASK_REGISTRY = {
    "classification": {
        "metrics": [
            {"col": "test_acc", "header": "Acc (\\%)",  "scale": 100,  "higher_is_better": True},
            {"col": "test_f1",  "header": "F1 Score",   "scale": 100,  "higher_is_better": True},
        ],
        "primary_metric": "test_acc",  # Used for delta computation only
    },
    "regression": {
        "metrics": [
            {"col": "test_mae",       "header": "MAE",          "scale": 1, "higher_is_better": False},
            {"col": "test_pearson_r", "header": "Pearson $r$",  "scale": 1, "higher_is_better": True},
            {"col": "test_r2",        "header": "$R^2$",         "scale": 1, "higher_is_better": True},
        ],
        "primary_metric": "test_pearson_r",
    },
}

def get_nested_cfg(cfg, keys):
    """Try multiple dot-separated key paths, return first non-None value."""
    for key in keys:
        try:
            val = OmegaConf.select(cfg, key)
            if val is not None:
                return val
        except Exception:
            continue
    return None


def infer_task(metrics: dict) -> str:
    """Infer task type from which metric columns are present in results.csv."""
    if "test_acc" in metrics:
        return "classification"
    if "test_rmse" in metrics or "test_mae" in metrics:
        return "regression"
    return "classification"


def parse_tuning(frozen: bool, from_scratch: bool) -> str:
    """Map (frozen, from_scratch) flags to a tuning strategy label."""
    if frozen and not from_scratch:
        return "LP"
    if frozen and from_scratch:
        return "From-Scratch LP"
    if not frozen and from_scratch:
        return "From-Scratch"
    return "Full FT"



def get_norm(cfg):
    """
    Infer normalisation strategy from the training transforms list.
    Returns a human-readable string for the normalisation method.
    """
    try:
        transforms = cfg.data.transforms.train
        if transforms is None:
            return "No Norm"
        transform_names = [t._target_.split(".")[-1] for t in transforms]

        if "PerChannelZScoreNormalise" in transform_names:
            return "Per Channel Z-Score"
        elif "GlobalZScoreNormalise" in transform_names:
            return "Global Z-Score"
        elif "MinMaxNormalise" in transform_names:
            return "Min-Max"
        elif "MeanSubtraction" in transform_names:
            return "Mean Subtraction"
        else:
            return "Min-Max [-1, 1]"
    except Exception:
        return "Unknown"


def get_resample(cfg):
    """
    Infer whether resampling is applied from the training transforms list.
    Returns a human-readable string.
    """
    try:
        transforms = cfg.data.transforms.train
        if transforms is None:
            return "No Resample"
        transform_names = [t._target_.split(".")[-1] for t in transforms]
        return "Resample" if "Resample" in transform_names else "No Resample"
    except Exception:
        return "Unknown"


def compile_benchmarks(exp_root: str = "./experiments") -> pd.DataFrame:
    """
    Iterate through all experiment directories in exp_root, read their config.yaml and results.csv files, and compile a DataFrame of metrics and configuration parameters.
    """

    root = Path(exp_root)
    all_data = []

    results_files = (
        list(root.glob("**/test/results.csv")) +
        list(root.glob("**/eval/results.csv"))
    )
    print(f"Found {len(results_files)} experiment results.")

    for res_path in results_files:
        cfg_path = res_path.parent / "config.yaml"
        if not cfg_path.exists():
            print(f"Skipping {res_path.parent.name}: no config.yaml found.")
            continue

        try:
            cfg = OmegaConf.load(cfg_path)
            metrics = pd.read_csv(res_path).iloc[0].to_dict()
            task = infer_task(metrics)

            window_duration = get_nested_cfg(cfg, ["data.window_duration"])
            if window_duration is None:
                window_length   = get_nested_cfg(cfg, ["data.window_length"])
                fs              = get_nested_cfg(cfg, ["data.sample_rate"])
                window_duration = window_length / fs if (window_length and fs) else None
            else:
                fs = get_nested_cfg(cfg, ["data.sample_rate"])
                window_length = int(window_duration * fs) if fs else None

            # model_name = get_nested_cfg(cfg, ["model.fm.name", "model.name"])
            model_name = get_nested_cfg(cfg, ["model.name", "model.fm.name"])
            model_name = model_name.replace("_fm", "").replace("_", "")

            split = get_nested_cfg(cfg, ["data.split_type", "task.split_type"])
            if split in ["subject", "user"]:
                split = "inter_subject"

            frozen = get_nested_cfg(cfg, ["task.freeze_backbone", "finetuning.freeze_backbone"])
            from_scratch = get_nested_cfg(cfg, ["task.from_scratch",    "finetuning.from_scratch"])

            record = {
                "exp_path":        str(res_path.parent.parent.name),
                "name":            get_nested_cfg(cfg, ["experiment.name"]),
                "seed":            get_nested_cfg(cfg, ["experiment.random_seed"]),
                "model_name":      model_name,
                "dataset":         get_nested_cfg(cfg, ["data.root_key"]),
                "window":          window_length,
                "fs":              fs,
                "window_duration": window_duration,
                "split":           split,
                "frozen":          frozen,
                "from_scratch":    from_scratch,
                "tuning":          parse_tuning(frozen, from_scratch),
                "task":            task,
                "pooling":         get_nested_cfg(cfg, ["model.wrapper.pooling_type", "task.pooling_type"]),
                "normalisation":   get_norm(cfg),
                "resample":        get_resample(cfg),
            }
            record.update(metrics)
            all_data.append(record)

        except Exception as e:
            print(f"Error processing {res_path}: {e}")

    return pd.DataFrame(all_data)


def _flag_best(summary: pd.DataFrame, mean_col: str, higher: bool) -> pd.DataFrame:
    """
    Add per-metric best-flagging columns to summary DataFrame.
    Flags the best value per (model, dataset) and overall per dataset.

    Args:
        summary:    aggregated DataFrame
        mean_col:   name of the mean column to flag (e.g. "acc_mean")
        higher:     True if higher values are better, False if lower

    Returns:
        summary with two new boolean columns:
            f"is_model_best_{mean_col}"
            f"is_overall_best_{mean_col}"
    """
    agg_fn = "max" if higher else "min"

    model_best   = summary.groupby(["model_name", "dataset"])[mean_col].transform(agg_fn)
    overall_best = summary.groupby("dataset")[mean_col].transform(agg_fn)

    summary[f"is_model_best_{mean_col}"]   = summary[mean_col] == model_best
    summary[f"is_overall_best_{mean_col}"] = summary[mean_col] == overall_best
    return summary

def safe_col(metric_col: str) -> str:
    """Strip 'test_' prefix for use as a DataFrame column name."""
    return metric_col.replace("test_", "")


def aggregate_seeds(df: pd.DataFrame, task: str = "classification") -> pd.DataFrame:
    task_cfg    = TASK_REGISTRY[task]
    metric_cfgs = task_cfg["metrics"]

    df = df[df["task"] == task].copy()

    # Build aggregation dict from available metric columns
    agg_dict = {"seed": ("seed", "count")}
    for mc in metric_cfgs:
        col = mc["col"]
        if col in df.columns:
            s = safe_col(col)
            agg_dict[f"{s}_mean"] = (col, "mean")
            agg_dict[f"{s}_std"]  = (col, "std")

    summary = df.groupby(
        ["model_name", "dataset", "split", "tuning", "window_duration", "task", "pooling", "normalisation", "resample"]
    ).agg(**agg_dict).reset_index()

    # Flag best per metric independently
    for mc in metric_cfgs:
        mean_col = f"{safe_col(mc['col'])}_mean"
        if mean_col in summary.columns:
            summary = _flag_best(summary, mean_col, mc["higher_is_better"])
    return summary


# ------------------------------------------------------------------ #
# Filtering
# ------------------------------------------------------------------ #
def refine_table(
    df: pd.DataFrame,
    task: str = None,
    split: str | None = None,
    window_duration: float = None,
    pooling: str | None = None,
) -> pd.DataFrame:
    """
    Filter aggregated summary to the datasets, split and window duration
    relevant for the paper table.

    Args:
        df: aggregated summary DataFrame
        task: "classification" or "regression"
        split: if provided, filter to this split type only
        window_duration: keep only rows with this window duration (seconds)
        pooling: if provided, filter to this pooling type only
    """
    dataset_map = DATASET_MAP[task]

    df = df.copy()
    df["model_name"]     = df["model_name"].map(lambda x: MODEL_MAP.get(x, x))
    df["dataset_pretty"] = df["dataset"].map(lambda x: dataset_map.get(x, x))

    known_datasets = set(dataset_map.values())
    mask = (df["dataset_pretty"].isin(known_datasets))

    if window_duration is not None:
        mask &= df["window_duration"] == window_duration

    if split is not None:
        mask &= df["split"] == split

    if pooling is not None:
        mask &= df["pooling"] == pooling

    return df[mask].copy()

# ------------------------------------------------------------------ #
# LaTeX formatting helpers
# ------------------------------------------------------------------ #
def _format_cell(
    mean: float,
    std: float,
    is_model_best: bool,
    is_overall_best: bool,
    scale: float = 1,
    decimals: int = 4,
) -> str:
    """
    Format a mean ± std cell with bold/underline hierarchy:
        overall best → \\textbf{\\underline{...}}
        model best   → \\textbf{...}
        otherwise    → plain
    """
    m = mean * scale
    s = std  * scale if not pd.isna(std) else 0.0
    dp = 2 if scale == 100 else decimals
    base = f"{m:.{dp}f} $\\pm$ {s:.{dp}f}"

    if is_overall_best:
        return f"\\textbf{{\\underline{{{base}}}}}"
    if is_model_best:
        return f"\\textbf{{{base}}}"
    return base


def _format_delta(val: float | None, is_best: bool) -> str:
    """Format a delta value with sign and optional bold."""
    if val is None:
        return "-"
    sign = "+" if val >= 0 else ""
    base = f"{sign}{val:.2f}"
    return f"\\textbf{{{base}}}" if is_best else base


def _latex_table_wrap(inner_lines: list[str], caption: str, label: str) -> list[str]:
    """Wrap table body lines in standard LaTeX table environment."""
    return (
        [r"\begin{table}[]", r"\centering", r"\resizebox{\columnwidth}{!}{%"]
        + inner_lines
        + [r"%", r"}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    )


def _cmidrules(n_datasets: int, start_col: int, span: int) -> str:
    """Generate a sequence of cmidrule commands for dataset column groups."""
    return "".join(
        "\\cmidrule(lr){" + f"{start_col + span*i}-{start_col + span*(i+1) - 1}" + "}"
        for i in range(n_datasets)
    )

# ------------------------------------------------------------------ #
# Main results table
# ------------------------------------------------------------------ #
def generate_paper_table(df: pd.DataFrame, 
                         task: str = "classification", 
                         split: str = None, 
                         pooling: str = None) -> str:
    
    df = refine_table(df, task=task, split=split, pooling=pooling)
    
    task_cfg    = TASK_REGISTRY[task]
    metric_cfgs = task_cfg["metrics"]
    n_metrics   = len(metric_cfgs)

    dataset_map = DATASET_MAP[task]
    datasets = [v for v in dataset_map.values() if v in df["dataset_pretty"].unique()]
    # Add window duration to dataset names
    datasets = [(ds, df[df["dataset_pretty"] == ds]["window_duration"].iloc[0]) for ds in datasets]
    models   = [m for m in MODEL_ORDER  if m in df["model_name"].unique()]
    tunings  = [t for t in TUNING_ORDER if t in df["tuning"].unique()]

    # Pre-compute formatted cells — one fmt column per metric
    df = df.copy()
    for mc in metric_cfgs:
        col      = safe_col(mc["col"])
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"
        fmt_col  = f"fmt_{col}"
        model_best_col   = f"is_model_best_{mean_col}"
        overall_best_col = f"is_overall_best_{mean_col}"

        if mean_col not in df.columns:
            df[fmt_col] = "-"
            continue

        df[fmt_col] = [
            _format_cell(
                r[mean_col],
                r.get(std_col, float("nan")),
                r.get(model_best_col,   False),
                r.get(overall_best_col, False),
                scale=mc["scale"],
            )
            for _, r in df.iterrows()
        ]

    # Build LaTeX lines
    inner = []
    inner.append(r"\begin{tabular}{ll" + "c" * n_metrics * len(datasets) + "}")
    inner.append(r"\toprule")

    # Header row 1: dataset names
    row1 = " & "
    for ds, dur in datasets:
        row1 += f" & \\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{ds} ({dur:.1f}s)}}}}"
    inner.append(row1 + r" \\")

    # Cmidrules + metric headers
    inner.append(_cmidrules(len(datasets), start_col=3, span=n_metrics))
    row2 = "Model & Tuning"
    for _ in datasets:
        for mc in metric_cfgs:
            row2 += f" & {mc['header']}"
    inner.append(row2 + r" \\")
    inner.append(r"\midrule")

    # Body
    for m_idx, model in enumerate(models):
        for tuning in tunings:
            row = f"{model} & {tuning}"
            for ds, dur in datasets:
                match = df[
                    (df["model_name"]     == model)  &
                    (df["tuning"]         == tuning) &
                    (df["dataset_pretty"] == ds)     &
                    (df["window_duration"] == dur)
                ]
                for mc in metric_cfgs:
                    fmt_col = f"fmt_{safe_col(mc['col'])}"
                    row += f" & {match.iloc[0][fmt_col]}" if not match.empty else " & -"
            inner.append(row + r" \\")

        if m_idx < len(models) - 1:
            inner.append(r"\midrule")

    inner.append(r"\bottomrule")
    inner.append(r"\end{tabular}")
    caption = f"Task: {task}, Split: {split}, Pooling: {pooling}" if split or pooling else f"Task: {task}"
    lines = _latex_table_wrap(inner, caption=caption, label="tab:results_" + task)
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Delta table
# ------------------------------------------------------------------ #
def compute_deltas(df: pd.DataFrame, 
                   task: str = "classification",
                   split: str | None = None,
                   pooling: str | None = None) -> pd.DataFrame:
    """
    Compute pretraining benefit deltas per model/dataset:
        ft_delta:   Full FT primary metric - From-Scratch FT primary metric
        head_delta: LP primary metric      - From-Scratch LP primary metric

    Deltas are always reported such that positive = pretraining helped,
    regardless of whether the primary metric is higher-is-better or not.
    """
    df = refine_table(df, task=task, split=split, pooling=pooling)

    task_cfg = TASK_REGISTRY[task]
    primary  = task_cfg["primary_metric"]
    higher   = next(
        mc["higher_is_better"]
        for mc in task_cfg["metrics"]
        if mc["col"] == primary
    )
    mean_col = f"{safe_col(primary)}_mean"

    records = []
    for model in df["model_name"].unique():
        for dataset in df["dataset_pretty"].unique():
            subset = df[
                (df["model_name"]     == model) &
                (df["dataset_pretty"] == dataset)
            ]

            def get_val(tuning: str) -> float | None:
                row = subset[subset["tuning"] == tuning]
                return float(row.iloc[0][mean_col]) if (not row.empty and mean_col in row.columns) else None

            full_ft      = get_val("Full FT")
            from_scratch = get_val("From-Scratch FT")
            lp           = get_val("LP")
            scratch_lp   = get_val("From-Scratch LP")

            def delta(a, b):
                if a is None or b is None:
                    return None
                d = (a - b) * 100
                return d if higher else -d  # flip so positive always = pretraining helped

            ft_delta   = delta(full_ft, from_scratch)
            head_delta = delta(lp, scratch_lp)

            if any(v is not None for v in [ft_delta, head_delta]):
                records.append({
                    "model_name":     model,
                    "dataset_pretty": dataset,
                    "ft_delta":       ft_delta,
                    "head_delta":     head_delta,
                })

    delta_df = pd.DataFrame(records)

    # Flag best per dataset (highest delta = most pretraining benefit)
    for col in ["ft_delta", "head_delta"]:
        if col in delta_df.columns:
            delta_df[f"{col}_is_best"] = (
                delta_df.groupby("dataset_pretty")[col].transform("max") == delta_df[col]
            )

    return delta_df


def generate_delta_table(delta_df: pd.DataFrame, 
                         task: str = "classification", 
                         split: str | None = None,
                         pooling: str | None = None) -> str:
    task_cfg = TASK_REGISTRY[task]
    primary  = task_cfg["primary_metric"]
    primary_header = next(
        mc["header"] for mc in task_cfg["metrics"] if mc["col"] == primary
    )

    dataset_map = DATASET_MAP[task]
    datasets = [v for v in dataset_map.values() if v in delta_df["dataset_pretty"].unique()]
    models   = [m for m in MODEL_ORDER if m in delta_df["model_name"].unique()]

    caption = (
        f"Pretraining benefit: gain in {primary_header} from pretrained initialisation "
        r"over random initialisation. "
        r"\textit{Full FT $\Delta$} = Full FT $-$ From-Scratch FT; "
        r"\textit{LP $\Delta$} = LP $-$ From-Scratch LP. "
        r"Positive values indicate that pretraining improved performance. "
        r"Best per dataset is \textbf{bolded}."
        f"Task: {task}, Split: {split}, Pooling: {pooling}" if split or pooling else f" Task: {task}"
    )

    inner = []
    inner.append(r"\begin{tabular}{l" + "cc" * len(datasets) + "}")
    inner.append(r"\toprule")

    row1 = " "
    for ds in datasets:
        row1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{ds}}}}}"
    inner.append(row1 + r" \\")

    inner.append(_cmidrules(len(datasets), start_col=2, span=2))

    row2 = "\\textbf{Model}"
    for _ in datasets:
        row2 += r" & Full FT $\Delta$ & LP $\Delta$"
    inner.append(row2 + r" \\")
    inner.append(r"\midrule")

    for model in models:
        row = model
        for ds in datasets:
            match = delta_df[
                (delta_df["model_name"]     == model) &
                (delta_df["dataset_pretty"] == ds)
            ]
            if not match.empty:
                r = match.iloc[0]
                row += " & " + _format_delta(r["ft_delta"],   r.get("ft_delta_is_best",   False))
                row += " & " + _format_delta(r["head_delta"], r.get("head_delta_is_best", False))
            else:
                row += " & - & -"
        inner.append(row + r" \\")

    inner.append(r"\bottomrule")
    inner.append(r"\end{tabular}")

    lines = _latex_table_wrap(inner, caption=caption, label=f"tab:delta_{task}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #
def plot_benchmark_results(
    df: pd.DataFrame,
    task: str = "classification",
    pooling: str | None = None,
    split: str | None = None,
    output_dir: str = "figures",
) -> None:
    
    plot_df = refine_table(df, task=task, pooling=pooling, split=split)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    task_cfg    = TASK_REGISTRY[task]
    metric_cfgs = task_cfg["metrics"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    for mc in metric_cfgs:
        col = mc["col"]
        if col not in plot_df.columns:
            continue

        g = sns.catplot(
            data=plot_df,
            kind="bar",
            x="model_name",
            y=col,
            hue="tuning",
            hue_order=TUNING_ORDER,
            col="dataset_pretty",
            col_wrap=1,
            height=4,
            aspect=2.5,
            errorbar="sd",
            palette="viridis",
            edgecolor="black",
            alpha=0.8,
            order=MODEL_ORDER,
        )
        g.set_titles("{col_name}")
        g.set_axis_labels("", mc["header"])
        g.legend.set_title("Strategy")

        fpath = output_dir / f"benchmark_{task}_{split}_{pooling}_{safe_col(col)}.pdf"
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {fpath}")




# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="experiments/")
    parser.add_argument("--output_dir",  type=str, default="figures/")
    args = parser.parse_args()

    master_df = compile_benchmarks(args.results_dir)
    master_df.to_csv("master_results.csv", index=False)
    print("Master results saved to master_results.csv")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for task in master_df["task"].unique():
        if task not in ["classification", "regression"]:
            print(f"Unknown task type: {task}, skipping.")
            continue

        for split in master_df[master_df["task"] == task]["split"].unique():
            for pooling in master_df[master_df["task"] == task]["pooling"].unique():
                if pd.isna(pooling):
                    continue
                
                print(f"\n{'='*60}\n  Task: {task.upper()}, Split: {split}, Pooling: {pooling}\n{'='*60}")

                # Replace null values for methods which dont use pooling or split, so that they are included in every table
                table_df = master_df.copy()
                table_df["pooling"] = table_df["pooling"].fillna(pooling)
                table_df["split"]   = table_df["split"].fillna(split)

                agg_df = aggregate_seeds(table_df, task=task)

                # Set anything with null pooling to "None" for consistency
                agg_df["pooling"] = agg_df["pooling"].fillna(pooling)
                agg_df["split"]   = agg_df["split"].fillna(split)

                ref_df = refine_table(agg_df, task=task, split=split, pooling=pooling)
                ref_df.to_csv(f"results_summaries/refined_results_{task}_{split}_{pooling}.csv", index=False)
                print(f"Refined results saved to results_summaries/refined_results_{task}_{split}_{pooling}.csv")

                if ref_df.empty:
                    print(f"No results found for task: {task}, split: {split}, pooling: {pooling}, skipping.")
                    continue

                # Main results table
                latex_table = generate_paper_table(ref_df, task=task, split=split, pooling=pooling)
                table_path  = output_dir / f"final_table_{task}_{split}_{pooling}.tex"
                table_path.write_text(latex_table)
                # print(f"\n--- LaTeX Table ({task}, {split}, {pooling}) ---\n{latex_table}")
                print(f"Saved to {table_path}")

                # Delta table
                try:
                    delta_df    = compute_deltas(ref_df, task=task, split=split, pooling=pooling)
                    delta_table = generate_delta_table(delta_df, task=task, split=split, pooling=pooling)
                    delta_path  = output_dir / f"delta_table_{task}_{split}_{pooling}.tex"
                    delta_path.write_text(delta_table)
                    # print(f"\n--- Delta Table ({task}, {split}, {pooling}) ---\n{delta_table}")
                    print(f"Saved to {delta_path}")
                except Exception as e:
                    print(f"Error computing delta table for task {task}, split {split}, pooling {pooling}: {e}")

                # Plots
                plot_benchmark_results(table_df, task=task, split=split, pooling=pooling, output_dir=args.output_dir)