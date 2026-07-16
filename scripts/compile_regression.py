import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import logging
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

TUNING_ORDER = ["Full FT", "From-Scratch FT", "LP", "From-Scratch LP"]
MODEL_ORDER  = ["NeuroRVQ", "PhysioWave", "TinyMyo"]

# ------------------------------------------------------------------ #
# Task registry — defines which metrics to compile, aggregate and
# display for each task type.
# ------------------------------------------------------------------ #
TASK_REGISTRY = {
    "classification": {
        "metrics": ["test_acc", "test_f1"],
        "display": [
            {"col": "test_acc", "header": "Acc (\\%)",  "scale": 100},
            {"col": "test_f1",  "header": "F1 Score",   "scale": 100},
        ],
        "primary_metric": "test_acc",   # Used for delta computation + best flagging
        "higher_is_better": True,
    },
    "regression": {
        "metrics": ["test_rmse", "test_pearson_r", "test_r2"],
        "display": [
            {"col": "test_rmse",      "header": "RMSE",       "scale": 1},
            {"col": "test_pearson_r", "header": "Pearson $r$", "scale": 1},
            {"col": "test_r2",        "header": "$R^2$",       "scale": 1},
        ],
        "primary_metric": "test_pearson_r",
        "higher_is_better": True,
    },
}

DATASET_TASK_MAP = {
    "epn612":          "classification",
    "ninapro_db5":     "classification",   # overridden to "regression" for pose results
    "uci_emg":         "classification",
    "neuromotor_wrist": "regression",
    "ninapro_db2":     "regression",
}

# When a dataset appears under multiple tasks (e.g. ninapro_db5 for both
# gesture classification and pose regression), the task is inferred from
# which metrics are present in results.csv rather than the dataset name alone.
def infer_task(metrics: dict) -> str:
    if "test_acc" in metrics:
        return "classification"
    if "test_rmse" in metrics:
        return "regression"
    return "classification"  # fallback


# ------------------------------------------------------------------ #
# Shared helpers (unchanged from original)
# ------------------------------------------------------------------ #
def get_nested_cfg(cfg, keys):
    for key in keys:
        try:
            val = OmegaConf.select(cfg, key)
            if val is not None:
                return val
        except Exception:
            continue
    return None


# ------------------------------------------------------------------ #
# Compilation
# ------------------------------------------------------------------ #
def compile_benchmarks(exp_root="./experiments"):
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
            print(f"Skipping {res_path.parent.name}: No resolved config found.")
            continue

        try:
            cfg = OmegaConf.load(cfg_path)
            metrics_df = pd.read_csv(res_path)
            metrics = metrics_df.iloc[0].to_dict()

            task = infer_task(metrics)

            window_length = get_nested_cfg(cfg, ["data.window_length"])
            fs            = get_nested_cfg(cfg, ["data.sample_rate"])
            window_duration = (
                window_length / fs
                if window_length is not None and fs is not None
                else None
            )

            model_name = get_nested_cfg(cfg, ["model.fm.name", "model.name"])
            model_name = model_name.replace("_fm", "").replace("_", "")

            split = get_nested_cfg(cfg, ["data.split_type", "task.split_type"])
            if split in ["subject", "user"]:
                split = "inter_subject"

            frozen       = get_nested_cfg(cfg, ["task.freeze_backbone",  "finetuning.freeze_backbone"])
            from_scratch = get_nested_cfg(cfg, ["task.from_scratch",     "finetuning.from_scratch"])

            if   frozen is True  and from_scratch is False: tuning = "LP"
            elif frozen is True  and from_scratch is True:  tuning = "From-Scratch LP"
            elif frozen is False and from_scratch is True:  tuning = "From-Scratch FT"
            else:                                           tuning = "Full FT"

            record = {
                "exp_path":        str(res_path.parent.parent.name),
                "name":            get_nested_cfg(cfg, ["experiment.name"]),
                "seed":            get_nested_cfg(cfg, ["experiment.random_seed"]),
                "model_name":      model_name,
                "dataset":         get_nested_cfg(cfg, ["data.root_key"]),
                "window":          get_nested_cfg(cfg, ["data.window_length"]),
                "fs":              get_nested_cfg(cfg, ["data.sample_rate"]),
                "window_duration": window_duration,
                "split":           split,
                "frozen":          frozen,
                "from_scratch":    from_scratch,
                "tuning":          tuning,
                "task":            task,
                "pooling":         get_nested_cfg(cfg, ["model.wrapper.pooling_type", "task.pooling_type"]),
            }
            record.update(metrics)
            all_data.append(record)

        except Exception as e:
            print(f"Error processing {res_path}: {e}")

    return pd.DataFrame(all_data)


# ------------------------------------------------------------------ #
# Aggregation
# ------------------------------------------------------------------ #
def aggregate_seeds(df, task="classification"):
    task_cfg = TASK_REGISTRY[task]
    metric_cols = task_cfg["metrics"]
    primary     = task_cfg["primary_metric"]
    higher      = task_cfg["higher_is_better"]

    # Only keep rows for this task
    df = df[df["task"] == task].copy()

    # Build agg dict dynamically from available metric columns
    agg_dict = {"seed": ("seed", "count")}
    for col in metric_cols:
        if col in df.columns:
            safe = col.replace("test_", "")
            agg_dict[f"{safe}_mean"] = (col, "mean")
            agg_dict[f"{safe}_std"]  = (col, "std")

    summary = df.groupby(
        ["model_name", "dataset", "split", "tuning", "window_duration", "task"]
    ).agg(**agg_dict).reset_index()

    # Best flagging using primary metric
    primary_safe  = primary.replace("test_", "")
    primary_mean  = f"{primary_safe}_mean"

    if primary_mean in summary.columns:
        if higher:
            model_best   = summary.groupby(["model_name", "dataset"])[primary_mean].transform("max")
            overall_best = summary.groupby("dataset")[primary_mean].transform("max")
        else:
            model_best   = summary.groupby(["model_name", "dataset"])[primary_mean].transform("min")
            overall_best = summary.groupby("dataset")[primary_mean].transform("min")

        summary["is_model_best"]   = summary[primary_mean] == model_best
        summary["is_overall_best"] = summary[primary_mean] == overall_best

    return summary


# ------------------------------------------------------------------ #
# Dataset / model name prettification
# ------------------------------------------------------------------ #
MODEL_MAP = {
    "tinymyo":    "TinyMyo",
    "neurorvq":   "NeuroRVQ",
    "physiowave": "PhysioWave",
}

DATASET_MAP_CLASSIFICATION = {
    "epn612":      "EPN-612 (1s)",
    "ninapro_db5": "NinaPro DB5 (1s)",
    "uci_emg":     "UCI-EMG (1s)",
}

DATASET_MAP_REGRESSION = {
    "ninapro_db5":      "NinaPro DB5 (1s)",
    "neuromotor_wrist": "Neuromotor-Wrist (1s)",
    "ninapro_db2":      "NinaPro DB2 (1s)",
}

SPLIT_MAP_CLASSIFICATION = "inter_subject"
SPLIT_MAP_REGRESSION     = "intra_session"   # default for regression; override if needed


def refine_table(df, task="classification"):
    dataset_map = (
        DATASET_MAP_CLASSIFICATION if task == "classification"
        else DATASET_MAP_REGRESSION
    )
    expected_split = (
        SPLIT_MAP_CLASSIFICATION if task == "classification"
        else SPLIT_MAP_REGRESSION
    )

    df = df.copy()
    df["model_name"]     = df["model_name"].map(lambda x: MODEL_MAP.get(x, x))
    df["dataset_pretty"] = df["dataset"].map(lambda x: dataset_map.get(x, x))

    # Filter to known datasets and correct split
    known_datasets = set(dataset_map.values())
    mask_dataset = df["dataset_pretty"].isin(known_datasets)
    mask_split   = df["split"] == expected_split
    mask_window  = df["window_duration"] == 1

    return df[mask_dataset & mask_split & mask_window].copy()


# ------------------------------------------------------------------ #
# LaTeX table generation
# ------------------------------------------------------------------ #
def _format_cell(mean, std, is_model_best, is_overall_best, scale=1, decimals=4):
    """Format a single mean ± std cell with optional bold/underline."""
    m = mean * scale
    s = std  * scale if not pd.isna(std) else 0.0

    # Choose decimal places based on scale
    dp = 2 if scale == 100 else decimals
    base = f"{m:.{dp}f} $\\pm$ {s:.{dp}f}"

    if is_overall_best:
        return f"\\textbf{{\\underline{{{base}}}}}"
    elif is_model_best:
        return f"\\textbf{{{base}}}"
    return base


def generate_paper_table(df, task="classification"):
    task_cfg  = TASK_REGISTRY[task]
    disp_cols = task_cfg["display"]
    n_metrics = len(disp_cols)

    dataset_map = (
        DATASET_MAP_CLASSIFICATION if task == "classification"
        else DATASET_MAP_REGRESSION
    )
    datasets = [v for v in dataset_map.values() if v in df["dataset_pretty"].unique()]
    models   = [m for m in MODEL_ORDER  if m in df["model_name"].unique()]
    tunings  = [t for t in TUNING_ORDER if t in df["tuning"].unique()]

    # Pre-compute formatted cells
    for dc in disp_cols:
        col      = dc["col"].replace("test_", "")
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"
        fmt_col  = f"fmt_{col}"
        scale    = dc["scale"]

        if mean_col not in df.columns:
            df[fmt_col] = "-"
            continue

        df[fmt_col] = [
            _format_cell(
                r[mean_col],
                r.get(std_col, float("nan")),
                r.get("is_model_best",   False),
                r.get("is_overall_best", False),
                scale=scale,
            )
            for _, r in df.iterrows()
        ]

    lines = []
    lines.append(r"\begin{table}[]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "c" * n_metrics * len(datasets) + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names
    row1 = " & "
    for ds in datasets:
        row1 += f" & \\multicolumn{{{n_metrics}}}{{c}}{{\\textbf{{{ds}}}}}"
    lines.append(row1 + r" \\")

    # Cmidrules
    cmidrules = "".join(
        "\\cmidrule(lr){" + f"{3 + n_metrics*i}-{2 + n_metrics*(i+1)}" + "}"
        for i in range(len(datasets))
    )
    lines.append(cmidrules)

    # Header row 2: metric names
    row2 = "Model & Tuning"
    for _ in datasets:
        for dc in disp_cols:
            row2 += f" & {dc['header']}"
    lines.append(row2 + r" \\")
    lines.append(r"\midrule")

    # Body
    for m_idx, model in enumerate(models):
        for tuning in tunings:
            row = f"{model} & {tuning}"
            for ds in datasets:
                match = df[
                    (df["model_name"]     == model)  &
                    (df["tuning"]         == tuning) &
                    (df["dataset_pretty"] == ds)
                ]
                for dc in disp_cols:
                    col     = dc["col"].replace("test_", "")
                    fmt_col = f"fmt_{col}"
                    if not match.empty and fmt_col in match.columns:
                        row += f" & {match.iloc[0][fmt_col]}"
                    else:
                        row += " & -"
            lines.append(row + r" \\")

        if m_idx < len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"%")
    lines.append(r"}")
    lines.append(r"\caption{Caption}")
    lines.append(r"\label{tab:placeholder}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Delta table
# ------------------------------------------------------------------ #
def compute_deltas(df, task="classification"):
    task_cfg = TASK_REGISTRY[task]
    primary  = task_cfg["primary_metric"].replace("test_", "")
    higher   = task_cfg["higher_is_better"]
    mean_col = f"{primary}_mean"

    records = []
    for model in df["model_name"].unique():
        for dataset in df["dataset_pretty"].unique():
            subset = df[
                (df["model_name"]     == model) &
                (df["dataset_pretty"] == dataset)
            ]

            def get_val(tuning):
                row = subset[subset["tuning"] == tuning]
                return row.iloc[0][mean_col] if (not row.empty and mean_col in row.columns) else None

            full_ft        = get_val("Full FT")
            from_scratch   = get_val("From-Scratch FT")
            lp             = get_val("LP")
            scratch_lp     = get_val("From-Scratch LP")

            ft_delta   = (full_ft - from_scratch) * 100 if (full_ft   is not None and from_scratch is not None) else None
            head_delta = (lp - scratch_lp)         * 100 if (lp        is not None and scratch_lp  is not None) else None

            # For regression metrics where lower is better (e.g. RMSE), flip sign
            if not higher:
                ft_delta   = -ft_delta   if ft_delta   is not None else None
                head_delta = -head_delta if head_delta is not None else None

            if any(v is not None for v in [ft_delta, head_delta]):
                records.append({
                    "model_name":     model,
                    "dataset_pretty": dataset,
                    "ft_delta":       ft_delta,
                    "head_delta":     head_delta,
                })

    delta_df = pd.DataFrame(records)

    for col in ["ft_delta", "head_delta"]:
        if col in delta_df.columns:
            delta_df[f"{col}_is_best"] = (
                delta_df.groupby("dataset_pretty")[col].transform("max") == delta_df[col]
            )

    return delta_df


def generate_delta_table(delta_df, task="classification"):
    task_cfg = TASK_REGISTRY[task]
    primary  = task_cfg["primary_metric"].replace("test_", "").replace("_", "\\_")
    dataset_map = (
        DATASET_MAP_CLASSIFICATION if task == "classification"
        else DATASET_MAP_REGRESSION
    )
    datasets = [v for v in dataset_map.values() if v in delta_df["dataset_pretty"].unique()]
    models   = [m for m in MODEL_ORDER if m in delta_df["model_name"].unique()]

    def format_delta(val, is_best):
        if val is None:
            return "-"
        sign = "+" if val >= 0 else ""
        base = f"{sign}{val:.2f}"
        return f"\\textbf{{{base}}}" if is_best else base

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Pretraining benefit: gain in " + primary +
        r" from pretrained initialisation over random initialisation. "
        r"\textit{Full FT $\Delta$} = Full FT $-$ From-Scratch FT; "
        r"\textit{LP $\Delta$} = LP $-$ From-Scratch LP. "
        r"Positive values indicate that pretraining improved performance. "
        r"Best per dataset is \textbf{bolded}.}"
    )
    lines.append(r"\label{tab:pretraining_delta_" + task + r"}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "cc" * len(datasets) + "}")
    lines.append(r"\toprule")

    row1 = " "
    for ds in datasets:
        row1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{ds}}}}}"
    lines.append(row1 + r" \\")

    cmidrules = "".join(
        "\\cmidrule(lr){" + f"{2+2*i}-{3+2*i}" + "}"
        for i in range(len(datasets))
    )
    lines.append(cmidrules)

    row2 = "\\textbf{Model}"
    for _ in datasets:
        row2 += r" & Full FT $\Delta$ & LP $\Delta$"
    lines.append(row2 + r" \\")
    lines.append(r"\midrule")

    for model in models:
        row = model
        for ds in datasets:
            match = delta_df[
                (delta_df["model_name"]     == model) &
                (delta_df["dataset_pretty"] == ds)
            ]
            if not match.empty:
                r = match.iloc[0]
                row += " & " + format_delta(r["ft_delta"],    r.get("ft_delta_is_best",   False))
                row += " & " + format_delta(r["head_delta"],  r.get("head_delta_is_best", False))
            else:
                row += " & - & -"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #
def plot_benchmark_results(df, task="classification", output_dir="figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    task_cfg = TASK_REGISTRY[task]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    plot_df = refine_table(df[df["task"] == task].copy(), task=task)

    for dc in task_cfg["display"]:
        col = dc["col"]
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
        g.set_axis_labels("", dc["header"])
        g.legend.set_title("Strategy")

        safe_col = col.replace("test_", "")
        fpath = output_dir / f"benchmark_{task}_{safe_col}.pdf"
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"Saved {col} plot to {fpath}")


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

    for task in ["classification", "regression"]:
        task_df = master_df[master_df["task"] == task]
        if task_df.empty:
            print(f"No results found for task: {task}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Task: {task.upper()}")
        print(f"{'='*60}")

        agg_df = aggregate_seeds(master_df, task=task)
        agg_df.to_csv(f"benchmark_summary_{task}.csv", index=False)
        print(agg_df)

        ref_df = refine_table(agg_df, task=task)

        # Main results table
        latex_table = generate_paper_table(ref_df, task=task)
        table_path  = output_dir / f"final_table_{task}.tex"
        table_path.write_text(latex_table)
        print(f"\n--- LaTeX Table ({task}) ---")
        print(latex_table)
        print(f"LaTeX table saved to {table_path}")

        # Delta table
        delta_df    = compute_deltas(ref_df, task=task)
        delta_table = generate_delta_table(delta_df, task=task)
        delta_path  = output_dir / f"delta_table_{task}.tex"
        delta_path.write_text(delta_table)
        print(f"\n--- Delta Table ({task}) ---")
        print(delta_table)
        print(f"Delta table saved to {delta_path}")

        # Plots
        plot_benchmark_results(master_df, task=task, output_dir=args.output_dir)