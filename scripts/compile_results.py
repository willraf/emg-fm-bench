import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import logging
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

def get_nested_cfg(cfg, keys):
    """Helper to try multiple keys and return the first that exists."""
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

    # Find all test results files
    results_files = list(root.glob("**/test/results.csv")) + list(root.glob("**/eval/results.csv"))
    print(f"Found {len(results_files)} experiment results.")

    for res_path in results_files:
        # 1. Path to the resolved config in the test folder
        cfg_path = res_path.parent / "config.yaml"
        
        if not cfg_path.exists():
            print(f"Skipping {res_path.parent.name}: No resolved config found.")
            continue

        try:
            cfg = OmegaConf.load(cfg_path)
            metrics_df = pd.read_csv(res_path)
            
            # Since test results usually only have one row
            metrics = metrics_df.iloc[0].to_dict()

            window_length = get_nested_cfg(cfg, ["data.window_length"])
            fs = get_nested_cfg(cfg, ["data.sample_rate"])
            window_duration = window_length / fs if window_length is not None and fs is not None else None

            # Multiple config keys and model renaming for backward compatibility
            model_name = get_nested_cfg(cfg, ["model.fm.name", "model.name"])
            model_name = model_name.replace("_fm", "").replace("_", "")
            split = get_nested_cfg(cfg, ["data.split_type", "task.split_type"])
            if split in ['subject', 'user']:
                split = 'inter_subject'

            frozen = get_nested_cfg(cfg, ["task.freeze_backbone", "finetuning.freeze_backbone"])
            from_scratch = get_nested_cfg(cfg, ["task.from_scratch", "finetuning.from_scratch"])
            if frozen is True and from_scratch is False:
                tuning = "Head"
            elif frozen is True and from_scratch is True:
                tuning = "Frozen (Scratch)"
            elif frozen is False and from_scratch is True:
                tuning = "From Scratch"
            else:
                tuning = "Full FT"

            # 2. Extract metadata with fallbacks
            record = {
                "exp_path": str(res_path.parent.parent.name),
                "name": get_nested_cfg(cfg, ["experiment.name"]),
                "seed": get_nested_cfg(cfg, ["experiment.random_seed"]),
                "model_name": model_name,
                "dataset": get_nested_cfg(cfg, ["data.root_key"]),
                "window": get_nested_cfg(cfg, ["data.window_length"]),
                "fs": get_nested_cfg(cfg, ["data.sample_rate"]),
                "window_duration": window_duration,
                "split": split,
                "frozen": get_nested_cfg(cfg, ["task.freeze_backbone", "finetuning.freeze_backbone"]),
                "from_scratch": get_nested_cfg(cfg, ["task.from_scratch", "finetuning.from_scratch"]),
                "tuning": tuning,
                "pooling": get_nested_cfg(cfg, ["model.wrapper.pooling_type", "task.pooling_type"]),
            }

            # 3. Merge metrics into the record
            record.update(metrics)
            all_data.append(record)

        except Exception as e:
            print(f"Error processing {res_path}: {e}")

    return pd.DataFrame(all_data)


def aggregate_seeds(df):
    """
    Compute mean and std of test accuracy and F1 score for each model-dataset-split-tuning combination,
    and identify the best performing model per dataset for bolding in the paper table.
    """
    df_refined = df.copy()

    # Take the mean across random seeds
    summary = df_refined.groupby(["model_name", "dataset", "split", "tuning", "window_duration"]).agg({
        "test_acc": ["mean", "std"],
        "test_f1": ["mean", "std"],
        "seed": "count"
    }).reset_index()

    summary.columns = ['model_name', 'dataset', 'split', 'tuning', 'window_duration', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std', 'seed_count']
    summary = summary.reset_index()

    # 4. Bold Best per Dataset (Comparing only Accuracy for the bolding logic)
    best_accs = summary.groupby('dataset')['acc_mean'].transform(max)
    summary['is_best_acc'] = summary['acc_mean'] == best_accs

    best_f1s = summary.groupby('dataset')['f1_mean'].transform(max)
    summary['is_best_f1'] = summary['f1_mean'] == best_f1s

    return summary


def refine_table(df):
    """
    Refine the benchmarking table to get just the results needed.
    """
    model_map = {
        "tinymyo": "TinyMyo",
        "neurorvq": "NeuroRVQ",
        "physiowave": "PhysioWave",
    }
    dataset_map = {
        "epn612": "EPN-612 (5s)",
        "ninapro_db5": "NinaPro DB5 (1s)",
        "uci_emg": "UCI-EMG (1s)"
    }

    df['model_name'] = df['model_name'].map(lambda x: model_map.get(x, x))
    df['dataset_pretty'] = df['dataset'].map(lambda x: dataset_map.get(x, x))
    
    # 2. Refine constraints
    mask_epn = (df['dataset'] == 'epn612') & (df['window_duration'] == 5)
    mask_nina = (df['dataset'] == 'ninapro_db5') & (df['window_duration'] == 1)
    mask_uci = (df['dataset'] == 'uci_emg') & (df['window_duration'] == 1)
    mask_split = (df['split'] == 'inter_subject')
    df_refined = df[(mask_epn | mask_nina | mask_uci) & mask_split].copy()

    return df_refined

def generate_paper_table(df):

    # Format Cells
    def format_cell(mean, std, is_best, is_acc=True):
        # Scale by 100 for Accuracy percentages
        m, s = (mean * 100, std * 100) #if is_acc else (mean, std)
        # Using standard size for SD, just with the \pm symbol
        base_str = f"{m:.2f} $\pm$ {s:.2f}" #if is_acc else f"{m:.3f} $\pm$ {s:.3f}"
        return f"\\textbf{{{base_str}}}" if is_best else base_str

    df['fmt_acc'] = [format_cell(r['acc_mean'], r['acc_std'], r['is_best_acc'], True) for _, r in df.iterrows()]
    df['fmt_f1'] = [format_cell(r['f1_mean'], r['f1_std'], r['is_best_f1'], False) for _, r in df.iterrows()]

    # Manual LaTeX Construction
    datasets = sorted(df['dataset_pretty'].unique())
    models = sorted(df['model_name'].unique())
    tunings = sorted(df['tuning'].unique())
    
    # Start with table, centering and resizing to fit column width
    header = "\\begin{table}[]\n\\centering\n\\resizebox{\\columnwidth}{!}{%\n"
    # Header with 2 columns per dataset
    header += "\\begin{tabular}{ll" + "cc" * len(datasets) + "}\n"
    header += "\\toprule\n"
    
    # Row 1: Merged Dataset Names
    header += " & " 
    for ds in datasets:
        header += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{ds}}}}}"
    header += " \\\\\n"
    
    # Row 2: Metric Names
    for i in range(len(datasets)):
        header += "\\cmidrule(lr){" + f"{3+2*i}-{4+2*i}" + "}"
    header += "Model & Tuning"
    for _ in datasets:
        header += " & Acc (\%) & F1 Score"
    header += " \\\\\n"
    header += "\\midrule\n"
    
    # Body
    body = ""
    # We iterate through models and tunings to build rows
    for model in models:
        for tuning in tunings:
            row_str = f"{model} & {tuning}"
            for ds in datasets:
                # Find the specific result for this combination
                match = df[(df['model_name'] == model) & 
                                (df['tuning'] == tuning) & 
                                (df['dataset_pretty'] == ds)]
                if not match.empty:
                    row_str += f" & {match.iloc[0]['fmt_acc']} & {match.iloc[0]['fmt_f1']}"
                else:
                    row_str += " & - & -"
            body += row_str + " \\\\\n"
        
    footer = "\\bottomrule\n\\end{tabular}"
    footer += "%\n}\n\caption{Caption}\n\\label{tab:placeholder}\n\\end{table}\n"
    
    return header + body + footer

def plot_benchmark_results(df, output_dir="figures"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set academic style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    df = refine_table(df)
    
    metrics = ["test_acc", "test_f1"]
    
    for metric in metrics:
        # Create a grid: Rows = Dataset, Cols = Metric
        g = sns.catplot(
            data=df, 
            kind="bar", 
            x="model_name", 
            y=metric,
            hue="tuning", 
            hue_order=["Frozen (Scratch)", "Head", "From Scratch", "Full FT"],
            col="dataset_pretty",
            col_wrap=1, 
            height=4, 
            aspect=2.5,
            errorbar='sd',
            palette="viridis",
            edgecolor="black",
            alpha=0.8
        )
        
        g.set_titles("{col_name}")
        g.set_axis_labels("", metric.upper().replace("TEST_", ""))
        g.legend.set_title("Strategy")
        
        # Save
        fpath = output_dir / f"benchmark_{metric}.pdf"
        plt.savefig(fpath, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric} plot to {fpath}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="experiments/")
    args = parser.parse_args()

    master_df = compile_benchmarks(args.results_dir)
    
    # Save the raw dataframe
    master_df.to_csv("master_results.csv", index=False)
    print("Master results saved to master_results.csv")

    agg_df = aggregate_seeds(master_df)
    print("\n--- Benchmarking Summary ---")
    print(agg_df)
    agg_df.to_csv("benchmark_summary.csv", index=False)


    ref_df = refine_table(agg_df)

    latex_table = generate_paper_table(ref_df)
    print("\n--- LaTeX Table ---")
    print(latex_table)

    plot_benchmark_results(master_df)