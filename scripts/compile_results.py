import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import logging
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

TUNING_ORDER = ["Full Fine-tuning", "From-Scratch Fine-tuning", "Linear Probing", "From-Scratch Linear Probing"]
MODEL_ORDER  = ["NeuroRVQ", "PhysioWave", "TinyMyo"]

def get_nested_cfg(cfg, keys):
    for key in keys:
        try:
            val = OmegaConf.select(cfg, key)
            if val is not None:
                return val
        except Exception:
            continue
    return None


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
            fs = get_nested_cfg(cfg, ["data.sample_rate"])
            window_duration = (
                window_length / fs
                if window_length is not None and fs is not None
                else None
            )

            model_name = get_nested_cfg(cfg, ["model.fm.name", "model.name"])
            model_name = model_name.replace("_fm", "").replace("_", "")

            split = get_nested_cfg(cfg, ["data.split_type", "task.split_type"])
            if split in ['subject', 'user']:
                split = 'inter_subject'

            frozen      = get_nested_cfg(cfg, ["task.freeze_backbone", "finetuning.freeze_backbone"])
            from_scratch = get_nested_cfg(cfg, ["task.from_scratch",    "finetuning.from_scratch"])

            if frozen is True and from_scratch is False:
                tuning = "Linear Probing"
            elif frozen is True and from_scratch is True:
                tuning = "From-Scratch Linear Probing"
            elif frozen is False and from_scratch is True:
                tuning = "From-Scratch Fine-tuning"
            else:
                tuning = "Full Fine-tuning"

            record = {
                "exp_path":       str(res_path.parent.parent.name),
                "name":           get_nested_cfg(cfg, ["experiment.name"]),
                "seed":           get_nested_cfg(cfg, ["experiment.random_seed"]),
                "model_name":     model_name,
                "dataset":        get_nested_cfg(cfg, ["data.root_key"]),
                "window":         get_nested_cfg(cfg, ["data.window_length"]),
                "fs":             get_nested_cfg(cfg, ["data.sample_rate"]),
                "window_duration": window_duration,
                "split":          split,
                "frozen":         frozen,
                "from_scratch":   from_scratch,
                "tuning":         tuning,
                "pooling":        get_nested_cfg(cfg, ["model.wrapper.pooling_type", "task.pooling_type"]),
            }
            record.update(metrics)
            all_data.append(record)

        except Exception as e:
            print(f"Error processing {res_path}: {e}")

    return pd.DataFrame(all_data)




def aggregate_seeds(df):
    summary = df.groupby(
        ["model_name", "dataset", "split", "tuning", "window_duration"]
    ).agg(
        acc_mean=("test_acc", "mean"),
        acc_std =("test_acc", "std"),
        f1_mean =("test_f1",  "mean"),
        f1_std  =("test_f1",  "std"),
        seed_count=("seed",   "count"),
    ).reset_index()

    # --- Per-model-per-dataset best (bold) ---
    model_best_acc = summary.groupby(["model_name", "dataset"])["acc_mean"].transform("max")
    model_best_f1  = summary.groupby(["model_name", "dataset"])["f1_mean"].transform("max")
    summary["is_model_best_acc"] = summary["acc_mean"] == model_best_acc
    summary["is_model_best_f1"]  = summary["f1_mean"]  == model_best_f1

    # --- Overall-per-dataset best (bold + underline) ---
    overall_best_acc = summary.groupby("dataset")["acc_mean"].transform("max")
    overall_best_f1  = summary.groupby("dataset")["f1_mean"].transform("max")
    summary["is_overall_best_acc"] = summary["acc_mean"] == overall_best_acc
    summary["is_overall_best_f1"]  = summary["f1_mean"]  == overall_best_f1

    return summary


def refine_table(df):
    model_map = {
        "tinymyo":    "TinyMyo",
        "neurorvq":   "NeuroRVQ",
        "physiowave": "PhysioWave",
    }
    dataset_map = {
        "epn612":       "EPN-612 (5s)",
        "ninapro_db5":  "NinaPro DB5 (1s)",
        "uci_emg":      "UCI-EMG (1s)",
    }

    df = df.copy()
    df["model_name"]     = df["model_name"].map(lambda x: model_map.get(x, x))
    df["dataset_pretty"] = df["dataset"].map(lambda x: dataset_map.get(x, x))

    mask_epn   = (df["dataset"] == "epn612")      & (df["window_duration"] == 5)
    mask_nina  = (df["dataset"] == "ninapro_db5") & (df["window_duration"] == 1)
    mask_uci   = (df["dataset"] == "uci_emg")     & (df["window_duration"] == 1)
    mask_split = (df["split"] == "inter_subject")

    return df[(mask_epn | mask_nina | mask_uci) & mask_split].copy()


def generate_paper_table(df):

    def format_cell(mean, std, is_model_best, is_overall_best):
        """
        Formatting hierarchy:
          overall best  → Bold Underline
          model best    → BOLD  
          otherwise     → xx.xx ± xx.xx
        """
        m, s = mean * 100, std * 100
        base = f"{m:.2f} $\\pm$ {s:.2f}"
        if is_overall_best:
            return f"\\textbf{{\\underline{{{base}}}}}"
        elif is_model_best:
            return f"\\textbf{{{base}}}"
        return base

    df = df.copy()
    df["fmt_acc"] = [
        format_cell(
            r["acc_mean"], r["acc_std"],
            r["is_model_best_acc"], r["is_overall_best_acc"]
        )
        for _, r in df.iterrows()
    ]
    df["fmt_f1"] = [
        format_cell(
            r["f1_mean"], r["f1_std"],
            r["is_model_best_f1"], r["is_overall_best_f1"]
        )
        for _, r in df.iterrows()
    ]

    # Use fixed meaningful orderings
    datasets = ["NinaPro DB5 (1s)", "UCI-EMG (1s)", "EPN-612 (5s)"]
    datasets = [d for d in datasets if d in df["dataset_pretty"].unique()]
    models   = [m for m in MODEL_ORDER  if m in df["model_name"].unique()]
    tunings  = [t for t in TUNING_ORDER if t in df["tuning"].unique()]

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    lines = []
    lines.append(r"\begin{table}[]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll" + "cc" * len(datasets) + "}")
    lines.append(r"\toprule")

    # Row 1: dataset names
    row1 = " & "
    for ds in datasets:
        row1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{ds}}}}}"
    lines.append(row1 + r" \\")

    # Row 2: cmidrules + metric headers
    cmidrules = "".join(
        "\\cmidrule(lr){" + f"{3+2*i}-{4+2*i}" + "}"
        for i in range(len(datasets))
    )
    lines.append(cmidrules)
    row2 = "Model & Tuning"
    for _ in datasets:
        row2 += " & Acc (\\%) & F1 Score"
    lines.append(row2 + r" \\")
    lines.append(r"\midrule")

    # ------------------------------------------------------------------ #
    # Body — midrule between model blocks
    # ------------------------------------------------------------------ #
    for m_idx, model in enumerate(models):
        for tuning in tunings:
            row = f"{model} & {tuning}"
            for ds in datasets:
                match = df[
                    (df["model_name"]     == model)  &
                    (df["tuning"]         == tuning) &
                    (df["dataset_pretty"] == ds)
                ]
                if not match.empty:
                    row += f" & {match.iloc[0]['fmt_acc']} & {match.iloc[0]['fmt_f1']}"
                else:
                    row += " & - & -"
            lines.append(row + r" \\")

        # Dividing line after each model block, except the last
        if m_idx < len(models) - 1:
            lines.append(r"\midrule")

    # ------------------------------------------------------------------ #
    # Footer
    # ------------------------------------------------------------------ #
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"%")
    lines.append(r"}")
    lines.append(r"\caption{Caption}")
    lines.append(r"\label{tab:placeholder}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def plot_benchmark_results(df, output_dir="figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    df = refine_table(df)

    for metric in ["test_acc", "test_f1"]:
        g = sns.catplot(
            data=df,
            kind="bar",
            x="model_name",
            y=metric,
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
        g.set_axis_labels("", metric.upper().replace("TEST_", ""))
        g.legend.set_title("Strategy")

        fpath = output_dir / f"benchmark_{metric}.pdf"
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"Saved {metric} plot to {fpath}")


def compute_deltas(df):

    delta_records = []

    models   = df["model_name"].unique()
    datasets = df["dataset_pretty"].unique()

    for model in models:
        for dataset in datasets:
            subset = df[
                (df["model_name"]     == model) &
                (df["dataset_pretty"] == dataset)
            ]




def compute_deltas(df):
    """
    Compute two pretraining benefit deltas per model/dataset:
        - ft_delta:   Full FT accuracy - From Scratch accuracy
        - head_delta: Head Only accuracy - Frozen Scratch accuracy

    A positive value means pretraining helped.

    Args:
        df: refined + aggregated dataframe (output of refine_table(aggregate_seeds(...)))

    Returns:
        DataFrame with columns:
            model_name, dataset_pretty, ft_delta, head_delta,
            ft_delta_f1, head_delta_f1
    """
    records = []

    models   = df["model_name"].unique()
    datasets = df["dataset_pretty"].unique()

    for model in models:
        for dataset in datasets:
            subset = df[
                (df["model_name"]     == model) &
                (df["dataset_pretty"] == dataset)
            ]

            def get_acc(tuning):
                row = subset[subset["tuning"] == tuning]
                return row.iloc[0]["acc_mean"] if not row.empty else None

            def get_f1(tuning):
                row = subset[subset["tuning"] == tuning]
                return row.iloc[0]["f1_mean"] if not row.empty else None

            full_ft       = get_acc("Full Fine-tuning")
            from_scratch  = get_acc("From-Scratch Fine-tuning")
            head          = get_acc("Linear Probing")
            frozen_scratch = get_acc("From-Scratch Linear Probing")

            full_ft_f1      = get_f1("Full Fine-tuning")
            from_scratch_f1 = get_f1("From-Scratch Fine-tuning")
            head_f1         = get_f1("Linear Probing")
            frozen_scratch_f1 = get_f1("From-Scratch Linear Probing")

            # Only include row if we have at least one delta
            if any(v is not None for v in [full_ft, from_scratch, head, frozen_scratch]):
                ft_delta   = (full_ft  - from_scratch)  * 100 if (full_ft  is not None and from_scratch  is not None) else None
                head_delta = (head - frozen_scratch) * 100 if (head is not None and frozen_scratch is not None) else None
                ft_delta_f1   = (full_ft_f1  - from_scratch_f1)  * 100 if (full_ft_f1  is not None and from_scratch_f1  is not None) else None
                head_delta_f1 = (head_f1 - frozen_scratch_f1) * 100 if (head_f1 is not None and frozen_scratch_f1 is not None) else None

                records.append({
                    "model_name":     model,
                    "dataset_pretty": dataset,
                    "ft_delta":       ft_delta,
                    "head_delta":     head_delta,
                    "ft_delta_f1":    ft_delta_f1,
                    "head_delta_f1":  head_delta_f1,
                })

    delta_df = pd.DataFrame(records)

    # Flag best per dataset for bolding
    for col in ["ft_delta", "head_delta"]:
        delta_df[f"{col}_is_best"] = (
            delta_df.groupby("dataset_pretty")[col].transform("max") == delta_df[col]
        )

    return delta_df


def generate_delta_table(delta_df):
    """
    Generate a LaTeX table showing pretraining benefit deltas.

    Columns: Model | Dataset | Full FT - Scratch (Acc) | Head - Frozen Scratch (Acc)
    Positive = pretraining helped. Negative = pretraining hurt.
    Best per dataset is bolded.
    """

    def format_delta(val, is_best):
        if val is None:
            return "-"
        sign = "+" if val >= 0 else ""
        base = f"{sign}{val:.2f}"
        return f"\\textbf{{{base}}}" if is_best else base

    # Fixed dataset ordering — consistent with main table
    datasets = ["NinaPro DB5 (1s)", "UCI-EMG (1s)", "EPN-612 (5s)"]
    datasets = [d for d in datasets if d in delta_df["dataset_pretty"].unique()]
    models   = [m for m in MODEL_ORDER if m in delta_df["model_name"].unique()]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pretraining benefit: accuracy gain (\%) from pretrained initialisaton "
                 r"over random initialisation. "
                 r"\textit{Full FT $\Delta$} = Full FT $-$ From Scratch; "
                 r"\textit{Head $\Delta$} = Head Only $-$ Frozen Random. "
                 r"Positive values indicate that pretraining improved performance. "
                 r"Best per dataset is \textbf{bolded}.}")
    lines.append(r"\label{tab:pretraining_delta}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")

    # Col spec: model | then per dataset: two cols
    lines.append(r"\begin{tabular}{l" + "cc" * len(datasets) + "}")
    lines.append(r"\toprule")

    # Row 1: dataset headers
    row1 = " "
    for ds in datasets:
        row1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{ds}}}}}"
    lines.append(row1 + r" \\")

    # Cmidrules
    cmidrules = "".join(
        "\\cmidrule(lr){" + f"{2+2*i}-{3+2*i}" + "}"
        for i in range(len(datasets))
    )
    lines.append(cmidrules)

    # Row 2: column headers
    row2 = "\\textbf{Model}"
    for _ in datasets:
        row2 += r" & Full FT $\Delta$ & Head $\Delta$"
    lines.append(row2 + r" \\")
    lines.append(r"\midrule")

    # Body
    for model in models:
        row = model
        for ds in datasets:
            match = delta_df[
                (delta_df["model_name"]     == model) &
                (delta_df["dataset_pretty"] == ds)
            ]
            if not match.empty:
                r = match.iloc[0]
                row += " & " + format_delta(r["ft_delta"],   r["ft_delta_is_best"])
                row += " & " + format_delta(r["head_delta"],  r["head_delta_is_best"])
            else:
                row += " & - & -"
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="experiments/")
    parser.add_argument("--output_dir",  type=str, default="figures/")
    args = parser.parse_args()

    master_df = compile_benchmarks(args.results_dir)
    master_df.to_csv("master_results.csv", index=False)
    print("Master results saved to master_results.csv")

    agg_df = aggregate_seeds(master_df)
    print("\n--- Benchmarking Summary ---")
    print(agg_df)
    agg_df.to_csv("benchmark_summary.csv", index=False)

    table_path = Path(args.output_dir) / "final_table.tex"
    ref_df = refine_table(agg_df)
    latex_table = generate_paper_table(ref_df)
    print("\n--- LaTeX Table ---")
    print(latex_table)
    table_path.write_text(latex_table)
    print(f"LaTeX table saved to {table_path}")


    # Delta table
    delta_path = Path(args.output_dir) / "delta_table.tex"
    delta_df = compute_deltas(ref_df)
    delta_table = generate_delta_table(delta_df)
    print("\n--- Delta Table ---")
    print(delta_table)
    delta_path.write_text(delta_table)
    print(f"Delta table saved to {delta_path}")


    plot_benchmark_results(master_df)