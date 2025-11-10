#!/usr/bin/env python3
"""
Analyze JDS (Just Dance Scores) correlation with motion metrics from CSV files.
Streamlined version - generates only essential plots and tables.

Output Plots:
- bias_hand_vs_arm_by_difficulty_cvpr.png
- bias_hand_vs_arm_by_difficulty_sidebyside.png
- validity_heatmap_hand_anonymous.png
- validity_summary_barplot_by_difficulty.png
- concordance_jds_hand_mean.png / median / max / last
- concordance_mpjpe_mean.png / median / max / last
- concordance_pa_mpjpe_mean.png / median / max / last
- concordance_dtw_mean.png / median / max / last

Output Tables:
- validity_correlation_table.tex (tab:validity)
- bias_by_difficulty.tex (tab:bias_difficulty)
- sensitivity_by_difficulty.tex (tab:sensitivity_difficulty)
- reliability_by_difficulty.tex (tab:reliability_jds)
- concordance_kendalls_w.tex (tab:concordance)

Usage:
    conda activate Switch4EAI
    python scripts/evaluate/analyze_jds_from_files_cleaned.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel
from scipy.stats import kendalltau
import pathlib
from glob import glob
import os
import sys

# Add parent directory to path
HERE = pathlib.Path(__file__).parent
sys.path.append(str(HERE / ".." / ".."))

# Import JDS scores dataset
from scripts.evaluate.data.jds_scores import data as jds_scores

# === Configuration ===
DEFAULT_HUMAN_RECORDINGS_GVHMR_BASE = pathlib.Path("/home/jkim3662/Videos/Switch4EAI/HumanRecordings_GVHMR/raw")

# Define color map for persons
PERSON_COLORS = {
    'WT': '#1f77b4', 'MG': '#ff7f0e', 'Ross': '#2ca02c', 'Nitish': '#d62728', 'Daesol': '#9467bd',
    'Rishi': '#8c564b', 'CH': '#e377c2', 'KY': '#7f7f7f', 'Mili': '#bcbd22', 'JH': '#17becf'
}

# Song information with levels
SONG_LEVELS = {
    "Old_Town_Road": 1,
    "Heart_Of_Glass": 2,
    "Unstoppable": 2,
    "Padam_Padam": 3,
    "Pink_Venom": 4
}


def load_all_csv_files(base_dir):
    """Discover and load all alignment_results.csv files."""
    csv_files = glob(str(base_dir / "**" / "alignment_results.csv"), recursive=True)
    all_data = []
    
    print(f"\nSearching for CSV files in: {base_dir}")
    print(f"Found {len(csv_files)} alignment_results.csv files")
    
    for csv_file in csv_files:
        try:
            # Path structure: .../Person/Song/Song_Person_condition/alignment_results.csv
            csv_path = pathlib.Path(csv_file)
            recording_dir = csv_path.parent.name  # e.g., "Old_Town_Road_WT_normal_1"
            
            # Get person and song from parent directories
            song_dir = csv_path.parent.parent.name  # e.g., "Old_Town_Road"
            person_dir = csv_path.parent.parent.parent.name  # e.g., "WT"
            
            # Extract condition from recording_dir
            # Standard format: {Song}_{Person}_{condition}
            expected_prefix = f"{song_dir}_{person_dir}_"
            condition_parts = recording_dir.replace(expected_prefix, "")
            
            # Load CSV
            df = pd.read_csv(csv_file)
            df['person'] = person_dir
            df['song'] = song_dir
            df['condition'] = condition_parts
            
            # Assign level and difficulty
            level = SONG_LEVELS.get(song_dir, 0)
            df['level'] = level
            df['difficulty'] = 'Easy' if level <= 2 else 'Hard'
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Warning: Could not parse {csv_file}: {e}")
            continue
    
    if len(all_data) == 0:
        raise ValueError("No CSV files found or could be parsed!")
    
    print(f"Successfully loaded {len(all_data)} CSV files")
    
    return pd.concat(all_data, ignore_index=True)


def merge_csv_with_jds(csv_df, jds_data):
    """Merge CSV data with JDS scores."""
    merged_data = []
    
    for _, row in csv_df.iterrows():
        person = row['person']
        song = row['song']
        condition = row['condition']
        
        jds_hand = None
        jds_arm = None
        
        if song in jds_data:
            if condition in jds_data[song]:
                jds_hand = jds_data[song][condition].get("Hand", {}).get(person)
                jds_arm = jds_data[song][condition].get("Arm", {}).get(person)
        
        merged_data.append({
            **row.to_dict(),
            'jds_hand': jds_hand,
            'jds_arm': jds_arm
        })
    
    return pd.DataFrame(merged_data)


def format_song_with_level(song_name):
    """Format song name with level information."""
    level_map = {
        'Old_Town_Road': 'Old Town Road (Lvl 1)',
        'Heart_Of_Glass': 'Heart Of Glass (Lvl 2)',
        'Unstoppable': 'Unstoppable (Lvl 2)',
        'Padam_Padam': 'Padam Padam (Lvl 3)',
        'Pink_Venom': 'Pink Venom (Lvl 4)'
    }
    return level_map.get(song_name, song_name)


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")

csv_data = load_all_csv_files(DEFAULT_HUMAN_RECORDINGS_GVHMR_BASE)
print(f"\nLoaded {len(csv_data)} records from CSV files")
print(f"Unique persons: {sorted(csv_data['person'].unique())}")
print(f"Unique songs: {sorted(csv_data['song'].unique())}")

merged_data = merge_csv_with_jds(csv_data, jds_scores)
merged_data_with_jds = merged_data[merged_data['jds_hand'].notna() | merged_data['jds_arm'].notna()].copy()
print(f"Loaded {len(merged_data_with_jds)} records with JDS scores")

# Add DTW metric
def compute_dtw_per_second(row):
    if pd.notna(row['pa_mpjpe_dtw']) and pd.notna(row['duration_sec']) and row['duration_sec'] > 0:
        return row['pa_mpjpe_dtw']
    return None

merged_data_with_jds['dtw'] = merged_data_with_jds.apply(compute_dtw_per_second, axis=1)

os.makedirs('plots', exist_ok=True)


# ============================================================================
# 1. VALIDITY: CORRELATION TABLE (tab:validity)
# ============================================================================
print("\n1. Generating validity correlation table...")

def calculate_validity():
    """Calculate validity (correlation between JDS and motion metrics) per song."""
    df = merged_data_with_jds.copy()
    df['run_type'] = df['condition'].str.split('_').str[0]
    df = df[df['run_type'] == 'normal'].copy()
    
    songs = sorted(df['song'].unique())
    metrics = ['dtw', 'mpjpe', 'pa_mpjpe']
    metric_names = {'dtw': 'DTW', 'mpjpe': 'MPJPE', 'pa_mpjpe': 'PA-MPJPE'}
    
    validity_results = []
    
    for song in songs:
        for metric_col in metrics:
            for jds_col, jds_name in [('jds_hand', 'Hand'), ('jds_arm', 'Arm')]:
                song_df = df[df['song'] == song]
                song_df = song_df[song_df[metric_col].notna() & song_df[jds_col].notna()]
                
                if len(song_df) < 3:
                    continue
                
                jds_vals = song_df[jds_col].values
                metric_vals = song_df[metric_col].values
                r, p = pearsonr(jds_vals, metric_vals)
                
                validity_results.append({
                    'Song': format_song_with_level(song),
                    'Metric': metric_names[metric_col],
                    'JDS_Type': jds_name,
                    'r': r,
                    'p': p,
                    'n': len(song_df)
                })
    
    return pd.DataFrame(validity_results)

validity_df = calculate_validity()

# Sort by song and metric order
song_order = ['Old Town Road (Lvl 1)', 'Heart Of Glass (Lvl 2)', 'Unstoppable (Lvl 2)', 
              'Padam Padam (Lvl 3)', 'Pink Venom (Lvl 4)']
metric_order = ['DTW', 'MPJPE', 'PA-MPJPE']

# Filter for Hand JDS only
validity_df_hand = validity_df[validity_df['JDS_Type'] == 'Hand'].copy()

# Map songs to anonymized names
song_anon_map = {
    'Old Town Road (Lvl 1)': 'Easy 1 (Lvl 1)',
    'Heart Of Glass (Lvl 2)': 'Easy 2 (Lvl 2)',
    'Unstoppable (Lvl 2)': 'Easy 3 (Lvl 2)',
    'Padam Padam (Lvl 3)': 'Hard 1 (Lvl 3)',
    'Pink Venom (Lvl 4)': 'Hard 2 (Lvl 4)'
}

# Generate LaTeX table with r and p values
with open('plots/validity_correlation_table.tex', 'w') as f:
    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n")
    f.write("\\small\n")
    f.write("\\resizebox{\\columnwidth}{!}{\%\n")
    f.write("\\begin{tabular}{@{}lcccccc@{}}\n")
    f.write("\\toprule\n")
    f.write("\\multirow{2}{*}{\\textbf{Song}} & \\multicolumn{2}{c}{\\textbf{DTW}} & \\multicolumn{2}{c}{\\textbf{MPJPE}} & \\multicolumn{2}{c}{\\textbf{PA-MPJPE}} \\\\\n")
    f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n")
    f.write(" & \\textbf{r} & \\textbf{p} & \\textbf{r} & \\textbf{p} & \\textbf{r} & \\textbf{p} \\\\\n")
    f.write("\\midrule\n")
    
    for song in song_order:
        song_data = validity_df_hand[validity_df_hand['Song'] == song]
        if len(song_data) > 0:
            anon_name = song_anon_map.get(song, song)
            row_vals = []
            for metric in metric_order:
                metric_data = song_data[song_data['Metric'] == metric]
                if len(metric_data) > 0:
                    r = metric_data.iloc[0]['r']
                    p = metric_data.iloc[0]['p']
                    if p < 0.001:
                        p_str = "$<$.001"
                    elif p < 0.01:
                        p_str = "$<$.01"
                    elif p < 0.05:
                        p_str = "$<$.05"
                    else:
                        p_str = f"{p:.2f}"
                    # Bold if significant
                    if p < 0.05:
                        row_vals.extend([f"\\textbf{{{r:.2f}}}", f"\\textbf{{{p_str}}}"])
                    else:
                        row_vals.extend([f"{r:.2f}", p_str])
                else:
                    row_vals.extend(["--", "--"])
            
            f.write(f"{anon_name} & {' & '.join(row_vals)} \\\\\n")
    
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("}\n\n")
    f.write("\\caption{Pearson correlation between Just Dance Scores (JDS) and conventional motion metrics with r and p values. Correlations computed per song. Values that show statistical significance ($p < 0.05$) are indicated in bold.}\n")
    f.write("\\label{tab:validity}\n")
    f.write("\\end{table}\n")

print("   ✓ validity_correlation_table.tex")


# ============================================================================
# 2. VALIDITY: HEATMAP (HAND, ANONYMOUS)
# ============================================================================
print("2. Generating validity heatmap...")

# Create heatmap data
heatmap_data = validity_df_hand.pivot_table(
    index='Metric',
    columns='Song',
    values='r',
    aggfunc='first'
)

# Reorder and rename
existing_songs = [s for s in song_order if s in heatmap_data.columns]
heatmap_data = heatmap_data[existing_songs]
existing_metrics = [m for m in metric_order if m in heatmap_data.index]
heatmap_data = heatmap_data.loc[existing_metrics]

# Rename columns to difficulty-based names
song_name_map = {
    'Old Town Road (Lvl 1)': 'Easy 1',
    'Heart Of Glass (Lvl 2)': 'Easy 2',
    'Unstoppable (Lvl 2)': 'Easy 3',
    'Padam Padam (Lvl 3)': 'Hard 1',
    'Pink Venom (Lvl 4)': 'Hard 2'
}
heatmap_data.columns = [song_name_map.get(col, col) for col in heatmap_data.columns]

# Create heatmap
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
    linewidths=0.5,
    linecolor='gray',
    ax=ax,
    annot_kws={'fontsize': 11, 'fontweight': 'bold'}
)

ax.set_title('Validity: Correlation between JDS and Motion Metrics',
            fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Song (by Difficulty)', fontsize=11, fontweight='bold')
ax.set_ylabel('Metric', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=0, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)

plt.tight_layout()
plt.savefig('plots/validity_heatmap_hand_anonymous.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ validity_heatmap_hand_anonymous.png")


# ============================================================================
# 3. VALIDITY: SUMMARY BARPLOT BY DIFFICULTY
# ============================================================================
print("3. Generating validity summary barplot...")
print(
"""
This is statistically problematic because:
Averaging correlation coefficients across different samples is not statistically sound
Averaging p-values is definitely incorrect - you can't meaningfully average p-values
The correct approach would be to pool all the data from songs within each difficulty level and then compute a single correlation
""")

def calculate_validity_by_difficulty_averaged():
    """Calculate validity (correlation) averaged by difficulty level."""
    df = merged_data_with_jds.copy()
    df['run_type'] = df['condition'].str.split('_').str[0]
    df = df[df['run_type'] == 'normal'].copy()
    
    songs = sorted(df['song'].unique())
    metric_names = ['DTW', 'MPJPE', 'PA-MPJPE']
    metric_cols = {'DTW': 'dtw', 'MPJPE': 'mpjpe', 'PA-MPJPE': 'pa_mpjpe'}
    
    song_validity = []
    
    for song in songs:
        song_df = df[df['song'] == song]
        difficulty = song_df['difficulty'].iloc[0] if len(song_df) > 0 else 'Unknown'
        
        for metric_name in metric_names:
            metric_col = metric_cols[metric_name]
            jds_col = 'jds_hand'
            jds_name = 'JDS (Hand)'
            
            valid_mask = song_df[jds_col].notna() & song_df[metric_col].notna()
            jds_vals = song_df.loc[valid_mask, jds_col].values
            motion_vals = song_df.loc[valid_mask, metric_col].values
            
            if len(jds_vals) >= 3:
                r, p = pearsonr(jds_vals, motion_vals)
                song_validity.append({
                    'Song': song,
                    'Difficulty': difficulty,
                    'JDS_Type': jds_name,
                    'Motion_Metric': metric_name,
                    'r': r,
                    'p': p
                })
    
    song_val_df = pd.DataFrame(song_validity)
    
    # Average within difficulty groups
    results = []
    for difficulty in ['Easy', 'Hard']:
        for jds_name in ['JDS (Hand)']:
            for metric_name in metric_names:
                diff_metric_df = song_val_df[
                    (song_val_df['Difficulty'] == difficulty) & 
                    (song_val_df['JDS_Type'] == jds_name) &
                    (song_val_df['Motion_Metric'] == metric_name)
                ]
                
                if len(diff_metric_df) > 0:
                    avg_r = diff_metric_df['r'].mean()
                    avg_p = diff_metric_df['p'].mean()
                    
                    results.append({
                        'Difficulty': difficulty,
                        'JDS_Type': jds_name,
                        'Motion_Metric': metric_name,
                        'r': avg_r,
                        'p': avg_p
                    })
    
    return pd.DataFrame(results)


    return pd.DataFrame(results)

validity_by_diff = calculate_validity_by_difficulty_averaged()
validity_hand_only = validity_by_diff[validity_by_diff['JDS_Type'] == 'JDS (Hand)'].copy()

# Prepare data for grouped bar plot
motion_metrics = ['DTW', 'MPJPE', 'PA-MPJPE']
difficulties = ['Easy', 'Hard']
colors_diff = {'Easy': '#2ecc71', 'Hard': '#e74c3c'}

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
x = np.arange(len(motion_metrics))
width = 0.35

# Plot bars for each difficulty
for i, difficulty in enumerate(difficulties):
    diff_data = validity_hand_only[validity_hand_only['Difficulty'] == difficulty]
    correlations = []
    
    for metric in motion_metrics:
        metric_row = diff_data[diff_data['Motion_Metric'] == metric]
        if len(metric_row) > 0:
            correlations.append(metric_row.iloc[0]['r'])
        else:
            correlations.append(0)
    
    offset = width * (i - 0.5)
    bars = ax.bar(x + offset, correlations, width, label=difficulty, 
                   color=colors_diff[difficulty], alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Pearson Correlation (r)', fontsize=12, fontweight='bold')
ax.set_xlabel('Motion Metric', fontsize=12, fontweight='bold')
ax.set_title('Validity: JDS Correlations with Motion Metrics by Difficulty', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(motion_metrics)
ax.legend(title='Difficulty', fontsize=11, title_fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('plots/validity_summary_barplot_by_difficulty.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ validity_summary_barplot_by_difficulty.png")


# ============================================================================
# 4. RELIABILITY: TABLE BY DIFFICULTY (tab:reliability_jds)
# ============================================================================
print("4. Generating reliability table...")

def calculate_icc(data):
    """Calculate ICC(3,1) and ICC(3,k) from repeated measurements."""
    n_subjects = len(data)
    n_raters = len(data[0]) if n_subjects > 0 else 0
    
    if n_subjects < 2 or n_raters < 2:
        raise ValueError("Not enough subjects or raters to calculate ICC.")
    
    data_array = np.array(data)
    
    # Calculate means
    grand_mean = np.mean(data_array)
    subject_means = np.mean(data_array, axis=1)
    rater_means = np.mean(data_array, axis=0)
    
    # Calculate sum of squares
    ss_total = np.sum((data_array - grand_mean) ** 2)
    ss_subjects = n_raters * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n_subjects * np.sum((rater_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters
    
    # Calculate mean squares
    ms_subjects = ss_subjects / (n_subjects - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
    
    # ICC(3,1) - single rater
    icc3_1 = (ms_subjects - ms_error) / (ms_subjects + (n_raters - 1) * ms_error)
    
    # ICC(3,k) - average of k raters
    icc3_k = (ms_subjects - ms_error) / ms_subjects
    
    return icc3_1, icc3_k


def calculate_reliability_by_difficulty_averaged():
    """Calculate test-retest reliability (ICC and CV) by difficulty level."""
    df = merged_data_with_jds.copy()
    df['run_type'] = df['condition'].str.split('_').str[0]
    df = df[df['run_type'] == 'normal'].copy()
    
    songs = sorted(df['song'].unique())
    metrics = [('jds_hand', 'JDS (Hand)')]
    
    song_reliability = []
    
    for song in songs:
        song_df = df[df['song'] == song]
        difficulty = song_df['difficulty'].iloc[0] if len(song_df) > 0 else 'Unknown'
        
        for metric_col, metric_name in metrics:
            # Group by person and get repeated measurements
            person_scores = {}
            for person in song_df['person'].unique():
                person_df = song_df[song_df['person'] == person]
                scores = person_df[metric_col].dropna().tolist()
                if len(scores) >= 2:  # Need at least 2 measurements
                    person_scores[person] = scores
            
            if len(person_scores) < 2:
                continue
            
            # Prepare data for ICC calculation (matrix: subjects x raters/trials)
            max_trials = max(len(v) for v in person_scores.values())
            icc_data = []
            for person, scores in person_scores.items():
                # Pad with NaN if needed (we'll handle this)
                padded = scores + [np.nan] * (max_trials - len(scores))
                icc_data.append(padded[:max_trials])
            
            # Filter out rows with NaN for ICC calculation
            icc_data_clean = []
            for row in icc_data:
                if not any(np.isnan(row)):
                    icc_data_clean.append(row)
            
            if len(icc_data_clean) >= 2:
                icc3_1, icc3_k = calculate_icc(icc_data_clean)
            else:
                icc3_1, icc3_k = np.nan, np.nan
            
            # Calculate CV (coefficient of variation)
            cv_values = []
            for person, scores in person_scores.items():
                if len(scores) >= 2:
                    mean_score = np.mean(scores)
                    std_score = np.std(scores, ddof=1)
                    if mean_score > 0:
                        cv = (std_score / mean_score) * 100
                        cv_values.append(cv)
            
            mean_cv = np.mean(cv_values) if len(cv_values) > 0 else np.nan
            
            song_reliability.append({
                'Song': song,
                'Difficulty': difficulty,
                'Metric': metric_name,
                'ICC(3,1)': icc3_1,
                'ICC(3,k)': icc3_k,
                'CV(%)': mean_cv
            })
    
    song_rel_df = pd.DataFrame(song_reliability)
    
    # Average within difficulty groups
    results = []
    for difficulty in ['Easy', 'Hard']:
        for metric_col, metric_name in metrics:
            diff_metric_df = song_rel_df[
                (song_rel_df['Difficulty'] == difficulty) & 
                (song_rel_df['Metric'] == metric_name)
            ]
            
            if len(diff_metric_df) > 0:
                avg_icc3_1 = diff_metric_df['ICC(3,1)'].mean()
                avg_icc3_k = diff_metric_df['ICC(3,k)'].mean()
                avg_cv = diff_metric_df['CV(%)'].mean()
                
                results.append({
                    'Difficulty': difficulty,
                    'Metric': metric_name,
                    'ICC(3,1)': avg_icc3_1,
                    'ICC(3,k)': avg_icc3_k,
                    'CV(%)': avg_cv
                })
    
    return pd.DataFrame(results)

reliability_by_diff = calculate_reliability_by_difficulty_averaged()
reliability_hand_by_diff = reliability_by_diff[reliability_by_diff['Metric'] == 'JDS (Hand)'].copy()

# Create LaTeX table for Hand JDS only
latex_lines = []
latex_lines.append("\\begin{table}[t]")
latex_lines.append("\\centering")
latex_lines.append("\\begin{tabular}{lccc}")
latex_lines.append("\\toprule")
latex_lines.append("\\textbf{Difficulty} & \\textbf{ICC(3,1)} & \\textbf{ICC(3,k)} & \\textbf{CV (\\%)} \\\\")
latex_lines.append("\\midrule")

for _, row in reliability_hand_by_diff.iterrows():
    difficulty_label = f"{row['Difficulty']} (Lvl {'1--2' if row['Difficulty'] == 'Easy' else '3--4'})"
    latex_lines.append(f"{difficulty_label} & {row['ICC(3,1)']:.2f} & {row['ICC(3,k)']:.2f} & {row['CV(%)']:.1f} \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\caption{Test--retest reliability of Just Dance Scores (JDS) averaged across songs within each difficulty level.}")
latex_lines.append("\\label{tab:reliability_jds}")
latex_lines.append("\\end{table}")

with open('plots/reliability_by_difficulty.tex', 'w') as f:
    f.write("\n".join(latex_lines))

print("   ✓ reliability_by_difficulty.tex")


# ============================================================================
# 5. SENSITIVITY: TABLE BY DIFFICULTY (tab:sensitivity_difficulty)
# ============================================================================
print("5. Generating sensitivity table...")

def calculate_sensitivity_by_difficulty_aggregated():
    """Calculate sensitivity (Normal vs Upperbody) by difficulty level."""
    df = merged_data_with_jds.copy()
    df['run_type'] = df['condition'].str.split('_').str[0]
    
    results = []
    
    for difficulty in ['Easy', 'Hard']:
        diff_df = df[df['difficulty'] == difficulty].copy()
        
        metric_col = 'jds_hand'
        metric_name = 'JDS (Hand)'
        
        # Get normal and upperbody values
        normal_df = diff_df[diff_df['run_type'] == 'normal']
        upper_df = diff_df[diff_df['run_type'] == 'upperbody']
        
        normal_vals = normal_df[metric_col].dropna().values
        upper_vals = upper_df[metric_col].dropna().values
        
        if len(normal_vals) >= 3 and len(upper_vals) >= 3:
            mean_normal = np.mean(normal_vals)
            mean_upper = np.mean(upper_vals)
            delta = mean_normal - mean_upper
            
            # Paired t-test (if we can pair by person and song)
            # For simplicity, use independent t-test
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(normal_vals, upper_vals)
            
            # Cohen's d
            pooled_std = np.sqrt((np.var(normal_vals, ddof=1) + np.var(upper_vals, ddof=1)) / 2)
            cohens_d = (mean_normal - mean_upper) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'Difficulty': difficulty,
                'Metric': metric_name,
                'Normal': mean_normal,
                'Upperbody': mean_upper,
                'Δ': delta,
                'p': p_val,
                'Cohens_d': cohens_d
            })
    
    return pd.DataFrame(results)

sensitivity_by_diff = calculate_sensitivity_by_difficulty_aggregated()
sensitivity_jds_by_diff = sensitivity_by_diff[sensitivity_by_diff['Metric'] == 'JDS (Hand)'].copy()

# Create LaTeX table
latex_lines = []
latex_lines.append("\\begin{table}[t]")
latex_lines.append("\\centering")
latex_lines.append("\\resizebox{\\columnwidth}{!}{")
latex_lines.append("\\begin{tabular}{l|cc|c|cc}")
latex_lines.append("\\toprule")
latex_lines.append(" & \\multicolumn{2}{c|}{\\textbf{Mean JDS (×10³)}} & \\textbf{Difference} & \\multicolumn{2}{c}{\\textbf{Statistics}} \\\\")
latex_lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){5-6}")
latex_lines.append("\\textbf{Difficulty} & Normal & Upperbody & $\\Delta$ & p & Cohen's d \\\\")
latex_lines.append("\\midrule")

for _, row in sensitivity_jds_by_diff.iterrows():
    difficulty = row['Difficulty']
    normal_val = row['Normal'] / 1000
    upper_val = row['Upperbody'] / 1000
    p_val = row['p']
    d_val = row['Cohens_d']
    
    # Round for display
    normal_display = round(normal_val, 1)
    upper_display = round(upper_val, 1)
    delta_display = normal_display - upper_display
    
    # Format delta with sign
    if delta_display > 0:
        delta_str = f"+{delta_display:.1f}"
    else:
        delta_str = f"{delta_display:.1f}"
    
    # Format p-value
    if p_val < 0.001:
        p_str = "$<$.001"
    elif p_val < 0.01:
        p_str = "$<$.01"
    elif p_val < 0.05:
        p_str = "$<$.05"
    else:
        p_str = f"{p_val:.2f}"
    
    difficulty_label = f"{difficulty} (Lvl {'1-2' if difficulty == 'Easy' else '3-4'})"
    latex_lines.append(f"{difficulty_label} & {normal_display} & {upper_display} & {delta_str} & {p_str} & {d_val:.2f} \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}}")
latex_lines.append("\\vspace{2mm}")
latex_lines.append("\\caption{Sensitivity of Just Dance Scores (JDS) to motion degradation grouped by difficulty level. All data within each difficulty level is aggregated/pooled before computing statistics. Values denote mean Just Dance Scores (×10³).}")
latex_lines.append("\\label{tab:sensitivity_difficulty}")
latex_lines.append("\\end{table}")

with open('plots/sensitivity_by_difficulty.tex', 'w') as f:
    f.write("\n".join(latex_lines))

print("   ✓ sensitivity_by_difficulty.tex")


# ============================================================================
# 6. BIAS: TABLE BY DIFFICULTY (tab:bias_difficulty)
# ============================================================================
print("6. Generating bias table...")

def calculate_bias_by_difficulty_aggregated():
    """Calculate Hand vs Arm bias by difficulty level."""
    df = merged_data_with_jds.copy()
    df['run_type'] = df['condition'].str.split('_').str[0]
    df = df[df['run_type'] == 'normal'].copy()
    
    results = []
    
    for difficulty in ['Easy', 'Hard']:
        diff_df = df[df['difficulty'] == difficulty].copy()
        
        hand_vals = diff_df['jds_hand'].dropna().values
        arm_vals = diff_df['jds_arm'].dropna().values
        
        if len(hand_vals) >= 3 and len(arm_vals) >= 3:
            mean_hand = np.mean(hand_vals)
            mean_arm = np.mean(arm_vals)
            delta = mean_hand - mean_arm
            delta_pct = (delta / mean_hand) * 100 if mean_hand > 0 else 0
            
            # Correlation between hand and arm
            valid_mask = diff_df['jds_hand'].notna() & diff_df['jds_arm'].notna()
            hand_paired = diff_df.loc[valid_mask, 'jds_hand'].values
            arm_paired = diff_df.loc[valid_mask, 'jds_arm'].values
            
            if len(hand_paired) >= 3:
                r, p = pearsonr(hand_paired, arm_paired)
            else:
                r, p = np.nan, np.nan
            
            results.append({
                'Difficulty': difficulty,
                'Mean_Hand': mean_hand,
                'Mean_Arm': mean_arm,
                'Δ': delta,
                'Δ(%)': delta_pct,
                'r': r,
                'p': p
            })
    
    return pd.DataFrame(results)

bias_by_diff = calculate_bias_by_difficulty_aggregated()

# Create LaTeX table
latex_lines = []
latex_lines.append("\\begin{table}[t]")
latex_lines.append("\\centering")
latex_lines.append("\\resizebox{\\columnwidth}{!}{")
latex_lines.append("\\begin{tabular}{l|cc|c|cc}")
latex_lines.append("\\toprule")
latex_lines.append(" & \\multicolumn{2}{c|}{\\textbf{Mean JDS (×10³)}} & \\textbf{Difference} & \\multicolumn{2}{c}{\\textbf{Statistics}} \\\\")
latex_lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){5-6}")
latex_lines.append("Difficulty & Hand & Arm & $\\Delta$(\\%) & r & p \\\\")
latex_lines.append("\\midrule")

for _, row in bias_by_diff.iterrows():
    hand_val = row['Mean_Hand'] / 1000
    arm_val = row['Mean_Arm'] / 1000
    delta_pct = row['Δ(%)']
    r_val = row['r']
    p_val = row['p']
    
    if p_val < 0.001:
        p_str = "$<$.001"
    elif p_val < 0.01:
        p_str = "$<$.01"
    elif p_val < 0.05:
        p_str = "$<$.05"
    else:
        p_str = f"{p_val:.2f}"
    
    difficulty_label = f"{row['Difficulty']} (Lvl {'1--2' if row['Difficulty'] == 'Easy' else '3--4'})"
    latex_lines.append(f"{difficulty_label} & {hand_val:.1f} & {arm_val:.1f} & {delta_pct:.1f} & {r_val:.2f} & {p_str} \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}}")
latex_lines.append("\\caption{Hand vs. Arm Just Dance Scores (JDS) Bias averaged by difficulty level.}")
latex_lines.append("\\label{tab:bias_difficulty}")
latex_lines.append("\\end{table}")

with open('plots/bias_by_difficulty.tex', 'w') as f:
    f.write("\n".join(latex_lines))

print("   ✓ bias_by_difficulty.tex")


# ============================================================================
# 7. BIAS: PLOT - CVPR VERSION
# ============================================================================
print("7. Generating bias plots...")
print("="*80)

df = merged_data_with_jds.copy()
df['run_type'] = df['condition'].str.split('_').str[0]
df = df[df['run_type'] == 'normal'].copy()
scatter_df = df[df['jds_hand'].notna() & df['jds_arm'].notna()].copy()

# ──────────────────────────────────────────────────────────────────────────
# VERSION 1: Combined scatter plot with difficulty coloring
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.4, 3.4))

# Plot points by difficulty
colors = {'Easy': 'skyblue', 'Hard': 'salmon'}
markers = {'Easy': 'o', 'Hard': 's'}

for difficulty in ['Easy', 'Hard']:
    diff_data = scatter_df[scatter_df['difficulty'] == difficulty]
    ax.scatter(
        diff_data['jds_hand'],
        diff_data['jds_arm'],
        c=colors[difficulty],
        marker=markers[difficulty],
        s=28,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.4,
        label=f'{difficulty} (Lvl {1 if difficulty == "Easy" else 3}-{2 if difficulty == "Easy" else 4})'
    )

# Overall regression line
hand_vals = scatter_df['jds_hand'].values
arm_vals = scatter_df['jds_arm'].values
r_overall, p_overall = pearsonr(hand_vals, arm_vals)
m, b = np.polyfit(hand_vals, arm_vals, 1)
xs = np.linspace(hand_vals.min(), hand_vals.max(), 100)
ax.plot(xs, m * xs + b, linestyle='--', color='black', linewidth=1.4,
        alpha=0.8, label=f'$r={r_overall:.2f}$, $p<.001$')

# Set equal axis scaling and rounded bounds
combined_min = min(scatter_df['jds_hand'].min(), scatter_df['jds_arm'].min())
combined_max = max(scatter_df['jds_hand'].max(), scatter_df['jds_arm'].max())
combined_min = np.floor(combined_min / 1000) * 1000
combined_max = np.ceil(combined_max / 1000) * 1000
ax.set_xlim(combined_min, combined_max)
ax.set_ylim(combined_min, combined_max)
ax.set_aspect('equal', adjustable='box')

# Dotted y=x reference line
ax.plot([combined_min, combined_max], [combined_min, combined_max],
        linestyle=':', color='gray', linewidth=1.2, alpha=0.8, label='y=x')

# Ticks: 3k, 5k, 7k, 9k, 11k, 13k
ticks = np.arange(3000, 14000, 2000)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.xaxis.set_major_formatter(lambda x, pos: f"{x/1000:.0f}k")
ax.yaxis.set_major_formatter(lambda y, pos: f"{y/1000:.0f}k")

# Labels and title
ax.set_xlabel('Hand JDS Score', fontsize=9, fontweight='bold')
ax.set_ylabel('Arm JDS Score', fontsize=9, fontweight='bold')
ax.set_title('Hand vs Arm JDS by Difficulty', fontsize=10, fontweight='bold')

# Legend and grid
ax.legend(fontsize=6.2, loc='upper left', framealpha=0.95, bbox_to_anchor=(0.02, 0.98))
ax.grid(True, linestyle='--', alpha=0.3)

# Print correlation stats for each difficulty
for difficulty in ['Easy', 'Hard']:
    diff_data = scatter_df[scatter_df['difficulty'] == difficulty]
    if len(diff_data) >= 3:
        r, p = pearsonr(diff_data['jds_hand'], diff_data['jds_arm'])
        print(f"  {difficulty}: r={r:.3f}, p={p:.4f}, n={len(diff_data)}")

# Tight layout for CVPR single-column fit
plt.tight_layout(pad=0.2)
plt.savefig('plots/bias_hand_vs_arm_by_difficulty_cvpr.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ bias_hand_vs_arm_by_difficulty_cvpr.png")


# ============================================================================
# 8. BIAS: PLOT - SIDE-BY-SIDE VERSION
# ============================================================================
# ──────────────────────────────────────────────────────────────────────────
# VERSION 2: Side-by-side subplots (Easy | Hard) for 2-column CVPR
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

difficulty_data = {
    'Easy': {'color': 'skyblue', 'marker': 'o'},
    'Hard': {'color': 'salmon', 'marker': 's'}
}

# Calculate global min/max across all data for consistent axes
global_min = min(scatter_df['jds_hand'].min(), scatter_df['jds_arm'].min())
global_max = max(scatter_df['jds_hand'].max(), scatter_df['jds_arm'].max())
global_min = np.floor(global_min / 1000) * 1000
global_max = np.ceil(global_max / 1000) * 1000

# Use fixed range: 3k to 13k
axis_min = 3000
axis_max = 13000

for idx, (difficulty, style) in enumerate(difficulty_data.items()):
    ax = axes[idx]
    diff_data = scatter_df[scatter_df['difficulty'] == difficulty]
    
    # Scatter plot
    ax.scatter(
        diff_data['jds_hand'],
        diff_data['jds_arm'],
        c=style['color'],
        marker=style['marker'],
        s=35,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Regression line
    if len(diff_data) >= 3:
        hand_sub = diff_data['jds_hand'].values
        arm_sub = diff_data['jds_arm'].values
        r_sub, p_sub = pearsonr(hand_sub, arm_sub)
        m_sub, b_sub = np.polyfit(hand_sub, arm_sub, 1)
        # Extend regression line across full axis range
        xs_sub = np.linspace(axis_min, axis_max, 100)
        ax.plot(xs_sub, m_sub * xs_sub + b_sub, linestyle='--', color='black', 
                linewidth=1.4, alpha=0.8, label=f'$r={r_sub:.2f}$, $p<.001$')
    
    # Set consistent axis scaling for both subplots
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    
    # y=x reference line across full range
    ax.plot([axis_min, axis_max], [axis_min, axis_max],
            linestyle=':', color='gray', linewidth=1.2, alpha=0.8, label='y=x')
    
    # Consistent ticks: 3k, 5k, 7k, 9k, 11k, 13k
    ticks_sub = np.arange(3000, 14000, 2000)
    ax.set_xticks(ticks_sub)
    ax.set_yticks(ticks_sub)
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x/1000:.0f}k")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y/1000:.0f}k")
    
    # Labels and title
    ax.set_xlabel('Hand JDS Score', fontsize=9, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Arm JDS Score', fontsize=9, fontweight='bold')
    
    level_range = '(Lvl 1-2)' if difficulty == 'Easy' else '(Lvl 3-4)'
    ax.set_title(f'{difficulty} Songs {level_range}', fontsize=10, fontweight='bold')
    
    # Legend and grid
    ax.legend(fontsize=7, loc='upper left', framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout(pad=0.5)
plt.savefig('plots/bias_hand_vs_arm_by_difficulty_sidebyside.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ bias_hand_vs_arm_by_difficulty_sidebyside.png")


# ============================================================================
# 9. CONCORDANCE: KENDALL'S W (tab:concordance)
# ============================================================================
print("9. Generating Kendall's W concordance analysis...")

def calculate_kendalls_w(rankings):
    """
    Calculate Kendall's W (coefficient of concordance) for a matrix of rankings.
    
    Args:
        rankings: 2D array where rows are judges/songs and columns are subjects/persons
    
    Returns:
        W: Kendall's coefficient of concordance
        chi_sq: Chi-square statistic
        p_value: p-value
    """
    rankings = np.array(rankings)
    n = rankings.shape[1]  # number of subjects (persons)
    m = rankings.shape[0]  # number of judges (songs)
    
    # Sum of ranks for each subject
    R_j = np.sum(rankings, axis=0)
    
    # Mean of rank sums
    R_bar = np.mean(R_j)
    
    # Sum of squared deviations
    S = np.sum((R_j - R_bar) ** 2)
    
    # Kendall's W
    W = (12 * S) / (m ** 2 * (n ** 3 - n))
    
    # Chi-square statistic
    chi_sq = m * (n - 1) * W
    
    # Degrees of freedom
    df = n - 1
    
    # P-value from chi-square distribution
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi_sq, df)
    
    return W, chi_sq, p_value


def prepare_concordance_data(aggregation='max', metric='jds_hand'):
    """
    Prepare data for concordance analysis.
    
    Args:
        aggregation: How to aggregate multiple trials ('mean', 'median', 'max', 'last')
        metric: Which metric to analyze ('jds_hand', 'mpjpe', 'pa_mpjpe', 'dtw')
    
    Returns:
        DataFrame with person rankings per song
    """
    df = merged_data_with_jds.copy()
    df['run_type'] = df['condition'].str.split('_').str[0]
    df = df[df['run_type'] == 'normal'].copy()
    
    # Filter to only include normal_1, normal_2, normal_3
    df['trial_num'] = df['condition'].str.extract(r'normal_(\d+)')[0].astype(float)
    df = df[df['trial_num'].isin([1, 2, 3])].copy()
    
    # Group by person and song, aggregate scores
    grouped = df.groupby(['person', 'song'])
    
    if aggregation == 'mean':
        agg_scores = grouped[metric].mean().reset_index()
    elif aggregation == 'median':
        agg_scores = grouped[metric].median().reset_index()
    elif aggregation == 'max':
        # For JDS: higher is better, for motion metrics: lower is better
        if metric == 'jds_hand':
            agg_scores = grouped[metric].max().reset_index()
        else:
            agg_scores = grouped[metric].min().reset_index()
    elif aggregation == 'last':
        # Get the last trial (highest trial_num)
        agg_scores = df.sort_values('trial_num').groupby(['person', 'song']).tail(1)[['person', 'song', metric]].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Pivot to get songs as rows and persons as columns
    pivot_scores = agg_scores.pivot(index='song', columns='person', values=metric)
    
    # Remove any rows or columns with all NaNs
    pivot_scores = pivot_scores.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Convert scores to rankings
    # For JDS: higher score = lower rank number (ascending=False)
    # For motion metrics: lower value = lower rank number (ascending=True)
    if metric == 'jds_hand':
        rankings = pivot_scores.rank(axis=1, ascending=False, method='average')
    else:
        rankings = pivot_scores.rank(axis=1, ascending=True, method='average')
    
    return rankings, pivot_scores


def calculate_concordance_all_methods(metric='jds_hand', metric_name='JDS (Hand)'):
    """Calculate Kendall's W for all aggregation methods for a given metric."""
    methods = ['mean', 'median', 'max', 'last']
    results = []
    
    print(f"   Calculating concordance for {metric_name}...")
    
    for method in methods:
        try:
            rankings, scores = prepare_concordance_data(aggregation=method, metric=metric)
            
            # Remove any persons with missing data across songs
            valid_persons = rankings.columns[rankings.notna().all()]
            rankings_clean = rankings[valid_persons]
            
            if len(rankings_clean) < 2 or len(valid_persons) < 3:
                print(f"      Warning: Not enough data for {method} aggregation")
                continue
            
            W, chi_sq, p_value = calculate_kendalls_w(rankings_clean.values)
            
            results.append({
                'Metric': metric_name,
                'Method': method.capitalize(),
                'Kendalls_W': W,
                'Chi_Square': chi_sq,
                'p_value': p_value,
                'n_songs': len(rankings_clean),
                'n_persons': len(valid_persons)
            })
            
            print(f"      {method.capitalize()}: W={W:.3f}, p={p_value:.4f}, n_songs={len(rankings_clean)}, n_persons={len(valid_persons)}")
            
        except Exception as e:
            print(f"      Error calculating concordance for {method}: {e}")
            continue
    
    return pd.DataFrame(results)


# Calculate concordance for all metrics
metrics_to_analyze = [
    ('jds_hand', 'JDS (Hand)'),
    ('mpjpe', 'MPJPE'),
    ('pa_mpjpe', 'PA-MPJPE'),
    ('dtw', 'DTW')
]

all_concordance_results = []
for metric, metric_name in metrics_to_analyze:
    concordance_results = calculate_concordance_all_methods(metric=metric, metric_name=metric_name)
    if len(concordance_results) > 0:
        all_concordance_results.append(concordance_results)

if len(all_concordance_results) > 0:
    combined_concordance = pd.concat(all_concordance_results, ignore_index=True)
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\resizebox{\\columnwidth}{!}{")
    latex_lines.append("\\begin{tabular}{llccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Metric} & \\textbf{Aggregation} & \\textbf{Kendall's W} & \\textbf{p-value} & \\textbf{n} \\\\")
    latex_lines.append("\\midrule")
    
    # For each aggregation method, find the best W across metrics
    aggregation_best = {}
    for method in ['Mean', 'Median', 'Max', 'Last']:
        method_data = combined_concordance[combined_concordance['Method'] == method]
        if len(method_data) > 0:
            aggregation_best[method] = method_data['Kendalls_W'].max()
    
    # Group by metric
    for metric_name in ['JDS (Hand)', 'MPJPE', 'PA-MPJPE', 'DTW']:
        metric_data = combined_concordance[combined_concordance['Metric'] == metric_name]
        if len(metric_data) == 0:
            continue
        
        for idx, (_, row) in enumerate(metric_data.iterrows()):
            method = row['Method']
            W = row['Kendalls_W']
            p = row['p_value']
            n_persons = int(row['n_persons'])
            
            # Format p-value
            if p < 0.001:
                p_str = "$<$.001"
            elif p < 0.01:
                p_str = "$<$.01"
            elif p < 0.05:
                p_str = "$<$.05"
            else:
                p_str = f"{p:.3f}"
            
            # Add metric name only for first row of each metric
            if idx == 0:
                metric_display = metric_name
            else:
                metric_display = ""
            
            # Bold if this is the best W for this aggregation method across all metrics
            if method in aggregation_best and W == aggregation_best[method]:
                latex_lines.append(f"{metric_display} & {method} & \\textbf{{{W:.3f}}} & {p_str} & {n_persons} \\\\")
            else:
                latex_lines.append(f"{metric_display} & {method} & {W:.3f} & {p_str} & {n_persons} \\\\")
        
        # Add separator between metrics (except after last metric)
        if metric_name != 'DTW':
            latex_lines.append("\\cmidrule(lr){1-5}")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}}")
    latex_lines.append("\\caption{Kendall's Coefficient of Concordance (W) showing agreement in player rankings across songs for different metrics. Four aggregation methods are shown for combining multiple trials (normal\\_1, normal\\_2, normal\\_3) per person-song pair. For JDS (Hand), higher scores are better; for motion metrics (MPJPE, PA-MPJPE, DTW), lower values are better. Values closer to 1 indicate perfect agreement in rankings. For each aggregation method, the best W value across metrics is shown in bold.}")
    latex_lines.append("\\label{tab:concordance}")
    latex_lines.append("\\end{table}")
    
    with open('plots/concordance_kendalls_w.tex', 'w') as f:
        f.write("\n".join(latex_lines))
    
    print("   ✓ concordance_kendalls_w.tex")
else:
    print("   ✗ No concordance results to save")


# ============================================================================
# 10. CONCORDANCE: VISUALIZATION (All aggregation methods for all metrics)
# ============================================================================
print("10. Generating concordance visualizations (all aggregation methods)...")

def create_concordance_plot(metric, metric_name, aggregation, filename, invert_for_normalization=False):
    """
    Create concordance visualization for a given metric and aggregation method.
    
    Args:
        metric: Metric column name
        metric_name: Display name for the metric
        aggregation: Aggregation method ('mean', 'median', 'max', 'last')
        filename: Output filename
        invert_for_normalization: If True, use (max - value) / max for normalization (for error metrics)
    """
    try:
        rankings_agg, scores_agg = prepare_concordance_data(aggregation=aggregation, metric=metric)
        
        # Remove any persons with missing data
        valid_persons = rankings_agg.columns[rankings_agg.notna().all()]
        rankings_clean = rankings_agg[valid_persons]
        scores_clean = scores_agg[valid_persons]
        
        if len(rankings_clean) >= 2 and len(valid_persons) >= 3:
            # Create a heatmap of rankings
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Rankings heatmap
            # Reorder songs by difficulty
            song_display_order = [s for s in ['Old_Town_Road', 'Heart_Of_Glass', 'Unstoppable', 
                                               'Padam_Padam', 'Pink_Venom'] if s in rankings_clean.index]
            rankings_ordered = rankings_clean.loc[song_display_order]
            
            # Map to display names
            rankings_ordered.index = [format_song_with_level(s) for s in rankings_ordered.index]
            
            sns.heatmap(
                rankings_ordered,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn_r',
                cbar_kws={'label': 'Rank (1=Best)', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray',
                ax=ax1,
                annot_kws={'fontsize': 9}
            )
            
            agg_display = aggregation.capitalize()
            if metric == 'jds_hand':
                rank_type = f"Best {agg_display} Score"
            else:
                rank_type = f"Best {agg_display} Value" if aggregation == 'max' else f"{agg_display} Value"
            ax1.set_title(f'Player Rankings Across Songs ({rank_type})', 
                         fontsize=12, fontweight='bold', pad=12)
            ax1.set_xlabel('Person', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Song', fontsize=11, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45, labelsize=9)
            ax1.tick_params(axis='y', rotation=0, labelsize=9)
            
            # Plot 2: Score consistency across songs (line plot)
            scores_ordered = scores_clean.loc[song_display_order]
            scores_ordered.index = [format_song_with_level(s) for s in song_display_order]
            
            # Normalize scores per song to show relative performance
            # Use min-max normalization for all metrics: (value - min) / (max - min) * 100
            # This gives 100 = best, 0 = worst for all metrics
            min_vals = scores_ordered.min(axis=1)
            max_vals = scores_ordered.max(axis=1)
            
            if invert_for_normalization:
                # For error metrics: lower is better, so invert
                # Best (minimum) gets 100, worst (maximum) gets 0
                scores_normalized = ((max_vals.values.reshape(-1, 1) - scores_ordered.values) / 
                                    (max_vals.values.reshape(-1, 1) - min_vals.values.reshape(-1, 1))) * 100
                scores_normalized = pd.DataFrame(scores_normalized, 
                                                index=scores_ordered.index, 
                                                columns=scores_ordered.columns)
            else:
                # For JDS: higher is better
                # Best (maximum) gets 100, worst (minimum) gets 0
                scores_normalized = ((scores_ordered.values - min_vals.values.reshape(-1, 1)) / 
                                    (max_vals.values.reshape(-1, 1) - min_vals.values.reshape(-1, 1))) * 100
                scores_normalized = pd.DataFrame(scores_normalized, 
                                                index=scores_ordered.index, 
                                                columns=scores_ordered.columns)
            
            for person in scores_normalized.columns:
                ax2.plot(range(len(scores_normalized)), 
                        scores_normalized[person].values,
                        marker='o', 
                        label=person,
                        linewidth=2,
                        markersize=8,
                        alpha=0.7)
            
            ax2.set_xticks(range(len(scores_normalized)))
            ax2.set_xticklabels(scores_normalized.index, rotation=45, ha='right', fontsize=9)
            
            # Use consistent label for all metrics
            ylabel = 'Normalized Performance (100 = Best, 0 = Worst)'
            ax2.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax2.set_xlabel('Song', fontsize=11, fontweight='bold')
            ax2.set_title(f'{metric_name} Consistency Across Songs', fontsize=12, fontweight='bold', pad=12)
            ax2.legend(loc='best', fontsize=8, ncol=2)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_ylim(0, 105)
            
            # Add Kendall's W annotation
            W, chi_sq, p_value = calculate_kendalls_w(rankings_clean.values)
            if p_value < 0.001:
                p_str = "p < .001"
            else:
                p_str = f"p = {p_value:.3f}"
            
            agg_label = aggregation.capitalize()
            fig.suptitle(f"Player Ranking Concordance - {metric_name} ({agg_label}) (Kendall's W = {W:.3f}, {p_str})",
                        fontsize=13, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            plt.savefig(f'plots/{filename}', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✓ {filename}")
        else:
            print(f"   ✗ Not enough data for {metric_name} ({aggregation}) concordance visualization")
            
    except Exception as e:
        print(f"   ✗ Error generating {metric_name} ({aggregation}) concordance visualization: {e}")


# Generate plots for all metrics and all aggregation methods
aggregation_methods = ['mean', 'median', 'max', 'last']
metric_configs = [
    ('jds_hand', 'JDS (Hand)', 'concordance_jds_hand', False),
    ('mpjpe', 'MPJPE', 'concordance_mpjpe', True),
    ('pa_mpjpe', 'PA-MPJPE', 'concordance_pa_mpjpe', True),
    ('dtw', 'DTW', 'concordance_dtw', True)
]

for metric, metric_name, base_filename, invert in metric_configs:
    for aggregation in aggregation_methods:
        filename = f"{base_filename}_{aggregation}.png"
        create_concordance_plot(metric, metric_name, aggregation, filename, invert)

print("\nDone! All plots and tables saved to plots/")
