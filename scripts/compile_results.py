import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import logging
import argparse

experiment_list = [
"experiments/2026-03-31_12-28-19_physio_wave_epn612_200_inter_subject_True_concat",
"experiments/2026-03-31_06-22-54_physio_wave_ninapro_db5_1000_intra_session_True_concat",
"experiments/2026-03-31_06-22-50_physio_wave_ninapro_db5_1000_inter_subject_False_concat",
"experiments/2026-03-31_06-22-45_physio_wave_ninapro_db5_1000_inter_subject_True_concat",
"experiments/2026-03-31_02-42-01_physio_wave_ninapro_db5_200_intra_session_False_concat",
"experiments/2026-03-30_23-21-12_physio_wave_ninapro_db5_200_intra_session_True_concat",
"experiments/2026-03-30_21-30-22_physio_wave_ninapro_db5_200_inter_subject_False_concat",
"experiments/2026-03-30_18-41-47_physio_wave_ninapro_db5_200_inter_subject_True_concat",
"experiments/2026-03-30_18-35-03_physio_wave_ninapro_db5_200_inter_subject_True_concat",
"experiments/2026-03-24_16-33-29_tiny_myo_epn612_1000_intra_session_False_concat",
"experiments/2026-03-24_13-38-04_tiny_myo_epn612_200_intra_session_False_concat",
"experiments/2026-03-24_12-03-12_tiny_myo_ninapro_db5_1000_intra_session_False_concat",
"experiments/2026-03-24_11-52-25_tiny_myo_ninapro_db5_200_intra_session_False_concat",
"experiments/2026-03-24_11-13-32_tiny_myo_ninapro_db5_1000_intra_session_False_concat",
"experiments/2026-03-24_10-58-01_tiny_myo_ninapro_db5_200_intra_session_False_concat",
"experiments/2026-03-24_10-49-16_tiny_myo_ninapro_db5_1000_intra_session_True_concat",
"experiments/2026-03-24_10-44-44_tiny_myo_ninapro_db5_200_intra_session_True_concat",
"experiments/2026-03-20_18-59-37_neuro_rvq_fm_epn612_1000_intra_session_True_concat",
"experiments/2026-03-20_13-38-34_neuro_rvq_fm_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-20_13-38-22_neuro_rvq_fm_epn612_1000_intra_session_True_concat",
"experiments/2026-03-20_13-38-02_neuro_rvq_fm_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-20_13-37-49_neuro_rvq_fm_epn612_1000_intra_session_True_concat",
"experiments/2026-03-20_13-37-28_neuro_rvq_fm_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-20_13-36-42_neuro_rvq_fm_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-20_13-36-26_neuro_rvq_fm_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-20_13-34-47_neuro_rvq_fm_epn612_1000_intra_session_True_concat",
"experiments/2026-03-20_13-34-25_neuro_rvq_fm_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-19_23-05-23_neuro_rvq_fm_ninapro_db5_1000_inter_subject_True_concat",
"experiments/2026-03-19_22-44-17_neuro_rvq_fm_ninapro_db5_1000_intra_session_True_concat",
"experiments/2026-03-19_18-39-12_neuro_rvq_fm_ninapro_db5_200_inter_subject_True_concat",
"experiments/2026-03-19_17-41-17_neuro_rvq_fm_ninapro_db5_200_intra_session_True_concat",
"experiments/2026-03-19_17-40-37_neuro_rvq_fm_ninapro_db5_1000_intra_session_True_concat",
"experiments/2026-03-19_17-40-37_neuro_rvq_fm_ninapro_db5_1000_inter_subject_True_concat",
"experiments/2026-03-19_17-40-37_neuro_rvq_fm_ninapro_db5_200_inter_subject_True_concat",
"experiments/2026-03-19_17-40-36_neuro_rvq_fm_ninapro_db5_200_intra_session_True_concat",
"experiments/2026-03-17_23-32-47_tiny_myo_epn612_1000_intra_session_True_concat",
"experiments/2026-03-17_08-09-15_tiny_myo_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-16_22-21-11_neuro_rvq_fm_epn612_1000_intra_session_True_concat",
"experiments/2026-03-16_18-15-10_neuro_rvq_fm_epn612_1000_inter_subject_True_concat",
"experiments/2026-03-16_14-19-48_neuro_rvq_fm_epn612_200_intra_session_True_concat",
"experiments/2026-03-16_12-33-43_neuro_rvq_fm_epn612_200_inter_subject_True_concat",
"experiments/2026-03-13_09-53-53_neuro_rvq_fm_ninapro_db5_1000_intra_session_False_concat",
"experiments/2026-03-13_09-31-07_neuro_rvq_fm_ninapro_db5_1000_subject_False_concat",
"experiments/2026-03-12_18-12-02_tiny_myo_ninapro_db5_1000_intra_session_False_concat",
"experiments/2026-03-12_18-01-33_neuro_rvq_fm_ninapro_db5_200_subject_True_concat",
"experiments/2026-03-12_17-02-50_neuro_rvq_fm_ninapro_db5_200_intra_session_True_concat",
"experiments/2026-03-12_10-07-25_tiny_myo_ninapro_db5_1000_intra_session_True_concat",
"experiments/2026-03-11_15-38-13_tiny_myo_ninapro_db5_1000s_subject_False_concat",
"experiments/2026-03-11_04-43-37_tinymyo_ninapro_db5_1000_subject_True_concat",
"experiments/2026-03-10_20-31-55_tinymyo_ninapro_db5_200_intra_session_False_concat",
"experiments/2026-03-10_17-39-52_tinymyo_ninapro_db5_200_intra_session_True_concat",
"experiments/2026-03-10_14-40-26_tinymyo_ninapro_db5_200_subject_False_concat",
"experiments/2026-03-10_12-00-03_tinymyo_ninapro_db5_200_subject_True_concat",
"experiments/2026-03-09_13-03-26_neurorvq_ninapro_db5_200s_intra_session_False_concat",
"experiments/2026-03-09_12-53-36_neurorvq_ninapro_db5_200s_subject_False_concat",
"experiments/2026-03-09_11-53-12_neurorvq_ninapro_db5_1000s_intra_session_True_concat",
"experiments/2026-03-09_11-45-29_neurorvq_ninapro_db5_1000s_subject_True_concat",
]

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
    results_files = list(root.glob("**/test/results.csv"))
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
            split = get_nested_cfg(cfg, ["data.split_type", "finetuning.split_type"])
            if split in ['subject', 'user']:
                split = 'inter_subject'

            frozen = get_nested_cfg(cfg, ["finetuning.freeze_backbone"])
            if frozen is True:
                tuning = "Head"
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
                "frozen": get_nested_cfg(cfg, ["finetuning.freeze_backbone"]),
                "tuning": tuning,
                "pooling": get_nested_cfg(cfg, ["model.wrapper.pooling_type", "finetuning.pooling_type"]),
            }

            # 3. Merge metrics into the record
            record.update(metrics)
            all_data.append(record)

        except Exception as e:
            print(f"Error processing {res_path}: {e}")

    return pd.DataFrame(all_data)

def benchmarking_table(df):
    # # 2. Refine constraints
    # mask_epn = (df['dataset'] == 'epn612') & (df['window_duration'] == 5)
    # mask_nina = (df['dataset'] == 'ninapro_db5') & (df['window_duration'] == 1)
    # mask_uci = (df['dataset'] == 'uci_emg') & (df['window_duration'] == 1)
    # mask_split = (df['split'] == 'inter_subject')
    # df_refined = df[(mask_epn | mask_nina | mask_uci) & mask_split].copy()
    df_refined = df.copy()

    # Take the mean across random seeds
    summary = df_refined.groupby(["model_name", "dataset", "split", "tuning", "window_duration"]).agg({
        "test_acc": ["mean", "std"],
        "test_f1": ["mean", "std"],
        "seed": "count"
    }).reset_index()

    summary.columns = ['model_name', 'dataset', 'split', 'tuning', 'window_duration', 'acc_mean', 'acc_std', 'f1_mean', 'f1_std', 'num_seeds']
    summary = summary.reset_index()

    # 4. Bold Best per Dataset (Comparing only Accuracy for the bolding logic)
    best_accs = summary.groupby('dataset')['acc_mean'].transform(max)
    summary['is_best_acc'] = summary['acc_mean'] == best_accs

    best_f1s = summary.groupby('dataset')['f1_mean'].transform(max)
    summary['is_best_f1'] = summary['f1_mean'] == best_f1s

    return summary

def generate_latex_table(df):

    # Clean up names for the table
    model_map = {
    "tinymyo": "TinyMyo",
    "neurorvq": "NeuroRVQ",
    "physiowave": "PhysioWave",
    }
    dataset_map = {
        "epn612": "EPN-612 (5s)",
        "ninapro_db5": "NinaPro DB5 (5s)",
        "uci_emg": "UCI-EMG (1s)"
    }
    
    df['model_name'] = df['model_name'].map(lambda x: model_map.get(x, x))
    df['dataset_pretty'] = df['dataset'].map(lambda x: dataset_map.get(x, x))

    # Format the metrics to be bold if they are the best for that dataset
    def format_cell(mean, std, is_best, is_acc=True):
        # Scale by 100 for Accuracy percentages
        m, s = (mean * 100, std * 100) if is_acc else (mean, std)
        # Using standard size for SD, just with the \pm symbol
        base_str = f"{m:.2f} $\pm$ {s:.2f}" if is_acc else f"{m:.3f} $\pm$ {s:.3f}"
        return f"\\textbf{{{base_str}}}" if is_best else base_str

    df['fmt_acc'] = [format_cell(r['acc_mean'], r['acc_std'], r['is_best_acc'], True) for _, r in df.iterrows()]
    df['fmt_f1'] = [format_cell(r['f1_mean'], r['f1_std'], r['is_best_f1'], False) for _, r in df.iterrows()]

    # Manual LaTeX Construction
    datasets = sorted(df['dataset_pretty'].unique())
    models = sorted(df['model_name'].unique())
    tunings = sorted(df['tuning'].unique())

    # Header with 2 columns per dataset
    header = "\\begin{tabular}{ll" + "cc" * len(datasets) + "}\n"
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
    
    return header + body + footer
    

def generate_paper_table(df):
    # 1. Name Mapping
    model_map = {
        "tinymyo": "TinyMyo",
        "neurorvq": "NeuroRVQ",
        "physiowave": "PhysioWave",
    }
    dataset_map = {
        "epn612": "EPN-612 (5s)",
        "ninapro_db5": "NinaPro DB5 (5s)",
        "uci_emg": "UCI-EMG (1s)"
    }

    print("\n--- Original Data ---")
    print(df)
    
    df['model_name'] = df['model_name'].map(lambda x: model_map.get(x, x))
    df['dataset_pretty'] = df['dataset'].map(lambda x: dataset_map.get(x, x))
    
    # 2. Refine constraints
    mask_epn = (df['dataset'] == 'epn612') & (df['window_duration'] == 5)
    mask_nina = (df['dataset'] == 'ninapro_db5') & (df['window_duration'] == 5)
    mask_uci = (df['dataset'] == 'uci_emg') & (df['window_duration'] == 1)
    mask_split = (df['split'] == 'inter_subject')
    df_refined = df[(mask_epn | mask_nina | mask_uci) & mask_split].copy()

    print("\n--- Refined Data ---")
    print(df_refined)

    # 3. Aggregate Seeds
    grouped = df_refined.groupby(["dataset_pretty", "model_name", "tuning"]).agg({
        "test_acc": ["mean", "std"],
        "test_f1": ["mean", "std"],
        "seed": "count"
    })

    grouped.columns = ['acc_mean', 'acc_std', 'f1_mean', 'f1_std', 'seed_count']
    grouped = grouped.reset_index()

    print("\n--- Grouped Results ---")
    print(grouped)

    # 4. Bold Best per Dataset (Comparing only Accuracy for the bolding logic)
    best_accs = grouped.groupby('dataset_pretty')['acc_mean'].transform(max)
    grouped['is_best_acc'] = grouped['acc_mean'] == best_accs

    best_f1s = grouped.groupby('dataset_pretty')['f1_mean'].transform(max)
    grouped['is_best_f1'] = grouped['f1_mean'] == best_f1s

    # 5. Format Cells
    def format_cell(mean, std, is_best, is_acc=True):
        # Scale by 100 for Accuracy percentages
        m, s = (mean * 100, std * 100) #if is_acc else (mean, std)
        # Using standard size for SD, just with the \pm symbol
        base_str = f"{m:.2f} $\pm$ {s:.2f}" #if is_acc else f"{m:.3f} $\pm$ {s:.3f}"
        return f"\\textbf{{{base_str}}}" if is_best else base_str

    grouped['fmt_acc'] = [format_cell(r['acc_mean'], r['acc_std'], r['is_best_acc'], True) for _, r in grouped.iterrows()]
    grouped['fmt_f1'] = [format_cell(r['f1_mean'], r['f1_std'], r['is_best_f1'], False) for _, r in grouped.iterrows()]

    # 6. Manual LaTeX Construction
    datasets = sorted(grouped['dataset_pretty'].unique())
    models = sorted(grouped['model_name'].unique())
    tunings = sorted(grouped['tuning'].unique())

    # Header with 2 columns per dataset
    header = "\\begin{tabular}{ll" + "cc" * len(datasets) + "}\n"
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
                match = grouped[(grouped['model_name'] == model) & 
                                (grouped['tuning'] == tuning) & 
                                (grouped['dataset_pretty'] == ds)]
                if not match.empty:
                    row_str += f" & {match.iloc[0]['fmt_acc']} & {match.iloc[0]['fmt_f1']}"
                else:
                    row_str += " & - & -"
            body += row_str + " \\\\\n"
        
    footer = "\\bottomrule\n\\end{tabular}"
    
    return header + body + footer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="experiments/")
    args = parser.parse_args()

    df = compile_benchmarks(args.results_dir)
    
    # Save the raw dataframe
    df.to_csv("master_results.csv", index=False)
    print("Master results saved to master_results.csv")

    # Example: Grouping by experiment type to see mean/std for your paper
    if not df.empty:
        # summary = df.groupby(["model_name", "dataset", "split", "frozen", "window"]).agg({
        #     "test_acc": ["mean", "std"],
        #     "test_f1": ["mean", "std"]
        # }).reset_index()
        
        # print("\n--- Summary Statistics (Mean ± Std) ---")
        # print(summary)
        # summary.to_csv("benchmark_summary.csv", index=False)

        summary = benchmarking_table(df)
        print("\n--- Benchmarking Summary ---")
        print(summary)
        summary.to_csv("benchmark_summary.csv", index=False)

        print("\n--- LaTeX Table ---")
        latex_table = generate_paper_table(df)
        print(latex_table)