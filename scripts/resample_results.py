import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_nested_cfg(cfg, keys):
    for key in keys:
        try:
            val = OmegaConf.select(cfg, key)
            if val is not None:
                return val
        except Exception:
            continue
    return None


def get_norm(cfg):
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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

            frozen       = get_nested_cfg(cfg, ["task.freeze_backbone", "finetuning.freeze_backbone"])
            from_scratch = get_nested_cfg(cfg, ["task.from_scratch",    "finetuning.from_scratch"])

            if frozen is True and from_scratch is False:
                tuning = "Head Only"
            elif frozen is True and from_scratch is True:
                tuning = "Frozen (Scratch)"
            elif frozen is False and from_scratch is True:
                tuning = "From Scratch"
            else:
                tuning = "Full FT"

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
                "tuning":          tuning,
                "pooling":         get_nested_cfg(cfg, ["model.wrapper.pooling_type", "task.pooling_type"]),
                "norm":            get_norm(cfg),
                "resample":        get_resample(cfg),
            }

            record.update(metrics)
            all_data.append(record)

        except Exception as e:
            print(f"Error processing {res_path}: {e}")

    return pd.DataFrame(all_data)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(df, group_cols):
    summary = (
        df.groupby(group_cols)
        .agg(
            acc_mean  =("test_acc", "mean"),
            acc_std   =("test_acc", "std"),
            f1_mean   =("test_f1",  "mean"),
            f1_std    =("test_f1",  "std"),
            seed_count=("seed",     "count"),
        )
        .reset_index()
    )
    summary["acc_std"] = summary["acc_std"].fillna(0)
    summary["f1_std"]  = summary["f1_std"].fillna(0)

    best_acc = summary.groupby("dataset")["acc_mean"].transform("max")
    best_f1  = summary.groupby("dataset")["f1_mean"].transform("max")
    summary["is_best_acc"] = summary["acc_mean"] == best_acc
    summary["is_best_f1"]  = summary["f1_mean"]  == best_f1

    return summary


# ---------------------------------------------------------------------------
# Resampling-specific filtering
# ---------------------------------------------------------------------------

RESAMPLE_ORDER = ["No Resample", "Resample"]

MODEL_MAP = {
    "tinymyo":    "TinyMyo",
    "neurorvq":   "NeuroRVQ",
    "physiowave": "PhysioWave",
}

DATASET_MAP = {
    "ninapro_db5": "NinaPro DB5",
    "uci_emg":     "UCI EMG",
}


def refine_resample_table(df):
    """
    Filter to the resampling ablation subset:
      - NinaPro DB5 and UCI EMG
      - Inter-subject / LOSO split
      - 1s windows
    """
    df = df.copy()
    df["model_name"]     = df["model_name"].map(lambda x: MODEL_MAP.get(x, x))
    df["dataset_pretty"] = df["dataset"].map(lambda x: DATASET_MAP.get(x, x))

    mask_nina = (
        (df["dataset"] == "ninapro_db5") &
        (df["split"].isin(["inter_subject", "loso"])) &
        (df["window_duration"].between(0.9, 1.1))
    )
    mask_uci = (
        (df["dataset"] == "uci_emg") &
        (df["split"].isin(["inter_subject", "loso"])) &
        (df["window_duration"].between(0.9, 1.1))
    )

    return df[mask_nina | mask_uci].copy()


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def format_cell(mean, std, is_best):
    m, s = mean * 100, std * 100
    base = f"{m:.2f} $\\pm$ {s:.2f}"
    return f"\\textbf{{{base}}}" if is_best else base


def generate_resample_table(df):
    """
    Generate a LaTeX table comparing resampling conditions.

    Rows:    Model × Tuning
    Columns: Resample condition × Metric (Acc, F1)
    """
    resample_vals = [r for r in RESAMPLE_ORDER if r in df["resample"].unique()]
    models  = sorted(df["model_name"].unique())
    tunings = sorted(df["tuning"].unique())

    n_cols = len(resample_vals)
    col_spec = "ll" + "cc" * n_cols

    lines = [
        "\\begin{table}[]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Row 1 — resample condition names
    row1 = "Model & Tuning"
    for r in resample_vals:
        row1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{r}}}}}"
    lines.append(row1 + " \\\\")

    # Cmidrule separators
    cmidrules = ""
    for i in range(n_cols):
        start = 3 + 2 * i
        end   = 4 + 2 * i
        cmidrules += f"\\cmidrule(lr){{{start}-{end}}}"
    lines.append(cmidrules)

    # Row 2 — metric names
    row2 = " & "
    for _ in resample_vals:
        row2 += " & Acc (\\%) & F1"
    lines.append(row2 + " \\\\")
    lines.append("\\midrule")

    # Body
    for model in models:
        first_model_row = True
        for tuning in tunings:
            row = f"{model if first_model_row else ''} & {tuning}"
            first_model_row = False

            for resample in resample_vals:
                match = df[
                    (df["model_name"] == model) &
                    (df["tuning"]     == tuning) &
                    (df["resample"]   == resample)
                ]
                if not match.empty:
                    r = match.iloc[0]
                    row += (
                        f" & {format_cell(r['acc_mean'], r['acc_std'], r['is_best_acc'])}"
                        f" & {format_cell(r['f1_mean'],  r['f1_std'],  r['is_best_f1'])}"
                    )
                else:
                    row += " & - & -"

            lines.append(row + " \\\\")
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"

    lines += [
        "\\end{tabular}",
        "%",
        "}",
        "\\caption{Effect of resampling on gesture classification accuracy and F1 score "
        "on NinaPro DB5 (1s windows, LOSO inter-subject evaluation).}",
        "\\label{tab:resample_ablation}",
        "\\end{table}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_resample_results(df, output_dir="figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    resample_order = [r for r in RESAMPLE_ORDER if r in df["resample"].unique()]
    model_order    = sorted(df["model_name"].unique())
    dataset_order  = [d for d in DATASET_MAP.values() if d in df["dataset_pretty"].unique()]

    for metric, ylabel in [("test_acc", "Accuracy"), ("test_f1", "F1 Score")]:
        g = sns.catplot(
            data=df,
            kind="bar",
            x="model_name",
            order=model_order,
            y=metric,
            hue="resample",
            hue_order=resample_order,
            col="tuning",
            row="dataset_pretty",
            row_order=dataset_order,
            height=3.5,
            aspect=1.6,
            errorbar="sd",
            palette="viridis",
            edgecolor="black",
            alpha=0.85,
            margin_titles=True,
            sharey=False,
        )

        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.set_axis_labels("Model", ylabel)
        g.legend.set_title("Resampling")

        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelrotation=0)

        fpath = output_dir / f"resample_ablation_{metric}.pdf"
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"Saved {fpath}")


def plot_resample_heatmap(df, output_dir="figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    resample_order = [r for r in RESAMPLE_ORDER if r in df["resample"].unique()]
    dataset_order  = [d for d in DATASET_MAP.values() if d in df["dataset_pretty"].unique()]

    n_datasets = len(dataset_order)
    fig, axes = plt.subplots(
        1, n_datasets,
        figsize=(len(resample_order) * 2.5 * n_datasets, len(df["model_name"].unique()) * 1.2 + 1),
        sharey=True,
    )
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, dataset_order):
        subset = df[df["dataset_pretty"] == dataset].copy()
        subset["model_tuning"] = subset["model_name"] + " — " + subset["tuning"]

        pivot = subset.pivot_table(
            index="model_tuning",
            columns="resample",
            values="acc_mean",
        )
        pivot = pivot[[r for r in resample_order if r in pivot.columns]] * 100

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Accuracy (%)"},
            vmin=0,
            vmax=100,
        )
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("Resampling")
        ax.set_ylabel("" if ax != axes[0] else "Model — Tuning")
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    fpath = output_dir / "resample_ablation_heatmap.pdf"
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"Saved {fpath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="experiments/")
    parser.add_argument("--output_dir",  type=str, default="figures/")
    args = parser.parse_args()

    master_df = compile_benchmarks(args.results_dir)
    master_df.to_csv("master_results_resample.csv", index=False)
    print(f"Master results: {len(master_df)} rows")

    resample_df = refine_resample_table(master_df)
    print(f"Resampling ablation subset: {len(resample_df)} rows")
    print(f"Resample values found: {resample_df['resample'].unique()}")
    print(f"Models found: {resample_df['model_name'].unique()}")

    group_cols = ["model_name", "dataset", "dataset_pretty", "split",
                  "tuning", "resample", "norm", "window_duration"]
    agg_df = aggregate_seeds(resample_df, group_cols)
    agg_df.to_csv("resample_ablation_summary.csv", index=False)
    print("\n--- Resampling Ablation Summary ---")
    print(agg_df[["model_name", "tuning", "resample", "norm", "acc_mean", "acc_std"]].to_string(index=False))

    latex = generate_resample_table(agg_df)
    print("\n--- LaTeX Table ---")
    print(latex)

    table_path = Path(args.output_dir) / "resample_ablation_table.tex"
    Path(args.output_dir).mkdir(exist_ok=True)
    table_path.write_text(latex)
    print(f"Table saved to {table_path}")

    plot_resample_results(resample_df, output_dir=args.output_dir)
    plot_resample_heatmap(agg_df,      output_dir=args.output_dir)