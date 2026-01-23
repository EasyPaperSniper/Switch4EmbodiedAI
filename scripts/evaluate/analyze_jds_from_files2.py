#!/usr/bin/env python3
"""
Analyze MPJPE from CSV files - compute overall statistics and best person.

Usage:
    conda activate Switch4EAI
    python scripts/evaluate/analyze_jds_from_files2.py
"""

import pandas as pd
import numpy as np
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

os.makedirs('plots', exist_ok=True)


# ============================================================================
# MPJPE ANALYSIS: Overall and Best Person
# ============================================================================
print("\n=== MPJPE Analysis ===")
print("\nCalculating MPJPE statistics...")

# Filter to only normal runs
df_normal = merged_data_with_jds.copy()
df_normal['run_type'] = df_normal['condition'].str.split('_').str[0]
df_normal = df_normal[df_normal['run_type'] == 'normal'].copy()

# Remove rows with missing MPJPE or PA-MPJPE
df_normal = df_normal[df_normal['mpjpe'].notna() & df_normal['pa_mpjpe'].notna()].copy()

# Convert MPJPE and PA-MPJPE from meters to millimeters
df_normal['mpjpe_mm'] = df_normal['mpjpe'] * 1000
df_normal['pa_mpjpe_mm'] = df_normal['pa_mpjpe'] * 1000

print(f"\nTotal records with MPJPE and PA-MPJPE (normal runs): {len(df_normal)}")

# Overall MPJPE statistics
print("\n--- Overall MPJPE Statistics (All People) ---")
overall_mean = df_normal['mpjpe_mm'].mean()
overall_median = df_normal['mpjpe_mm'].median()
overall_std = df_normal['mpjpe_mm'].std()
overall_min = df_normal['mpjpe_mm'].min()
overall_max = df_normal['mpjpe_mm'].max()

print(f"Mean MPJPE: {overall_mean:.2f} mm")
print(f"Median MPJPE: {overall_median:.2f} mm")
print(f"Std Dev: {overall_std:.2f} mm")
print(f"Min: {overall_min:.2f} mm")
print(f"Max: {overall_max:.2f} mm")

# Per-person MPJPE statistics
print("\n--- Per-Person MPJPE Statistics ---")
person_stats = df_normal.groupby('person')['mpjpe_mm'].agg(['mean', 'median', 'std', 'count']).round(2)
person_stats = person_stats.sort_values('mean')
print(person_stats)

# Best person (lowest mean MPJPE)
best_person = person_stats.index[0]
best_person_mean = person_stats.loc[best_person, 'mean']
best_person_median = person_stats.loc[best_person, 'median']
best_person_std = person_stats.loc[best_person, 'std']
best_person_count = int(person_stats.loc[best_person, 'count'])

print(f"\n--- Best Person (Lowest Mean MPJPE) ---")
print(f"Person: {best_person}")
print(f"Mean MPJPE: {best_person_mean:.2f} mm")
print(f"Median MPJPE: {best_person_median:.2f} mm")
print(f"Std Dev: {best_person_std:.2f} mm")
print(f"Number of trials: {best_person_count}")

# Per-song MPJPE statistics
print("\n--- Per-Song MPJPE Statistics ---")
song_stats = df_normal.groupby('song')['mpjpe_mm'].agg(['mean', 'median', 'std', 'count']).round(2)
song_stats = song_stats.sort_values('mean')
song_stats.index = [format_song_with_level(s) for s in song_stats.index]
print(song_stats)

# Per-difficulty MPJPE statistics
print("\n--- Per-Difficulty MPJPE Statistics ---")
diff_stats = df_normal.groupby('difficulty')['mpjpe_mm'].agg(['mean', 'median', 'std', 'count']).round(2)
print(diff_stats)


# ============================================================================
# PA-MPJPE ANALYSIS: Overall and Best Person
# ============================================================================
print("\n" + "=" * 80)
print("\n=== PA-MPJPE Analysis ===")
print("\nCalculating PA-MPJPE statistics...")

# Overall PA-MPJPE statistics
print("\n--- Overall PA-MPJPE Statistics (All People) ---")
overall_pa_mean = df_normal['pa_mpjpe_mm'].mean()
overall_pa_median = df_normal['pa_mpjpe_mm'].median()
overall_pa_std = df_normal['pa_mpjpe_mm'].std()
overall_pa_min = df_normal['pa_mpjpe_mm'].min()
overall_pa_max = df_normal['pa_mpjpe_mm'].max()

print(f"Mean PA-MPJPE: {overall_pa_mean:.2f} mm")
print(f"Median PA-MPJPE: {overall_pa_median:.2f} mm")
print(f"Std Dev: {overall_pa_std:.2f} mm")
print(f"Min: {overall_pa_min:.2f} mm")
print(f"Max: {overall_pa_max:.2f} mm")

# Per-person PA-MPJPE statistics
print("\n--- Per-Person PA-MPJPE Statistics ---")
person_pa_stats = df_normal.groupby('person')['pa_mpjpe_mm'].agg(['mean', 'median', 'std', 'count']).round(2)
person_pa_stats = person_pa_stats.sort_values('mean')
print(person_pa_stats)

# Best person (lowest mean PA-MPJPE)
best_person_pa = person_pa_stats.index[0]
best_person_pa_mean = person_pa_stats.loc[best_person_pa, 'mean']
best_person_pa_median = person_pa_stats.loc[best_person_pa, 'median']
best_person_pa_std = person_pa_stats.loc[best_person_pa, 'std']
best_person_pa_count = int(person_pa_stats.loc[best_person_pa, 'count'])

print(f"\n--- Best Person (Lowest Mean PA-MPJPE) ---")
print(f"Person: {best_person_pa}")
print(f"Mean PA-MPJPE: {best_person_pa_mean:.2f} mm")
print(f"Median PA-MPJPE: {best_person_pa_median:.2f} mm")
print(f"Std Dev: {best_person_pa_std:.2f} mm")
print(f"Number of trials: {best_person_pa_count}")

# Per-song PA-MPJPE statistics
print("\n--- Per-Song PA-MPJPE Statistics ---")
song_pa_stats = df_normal.groupby('song')['pa_mpjpe_mm'].agg(['mean', 'median', 'std', 'count']).round(2)
song_pa_stats = song_pa_stats.sort_values('mean')
song_pa_stats.index = [format_song_with_level(s) for s in song_pa_stats.index]
print(song_pa_stats)

# Per-difficulty PA-MPJPE statistics
print("\n--- Per-Difficulty PA-MPJPE Statistics ---")
diff_pa_stats = df_normal.groupby('difficulty')['pa_mpjpe_mm'].agg(['mean', 'median', 'std', 'count']).round(2)
print(diff_pa_stats)


# Save summary to file
summary_lines = []
summary_lines.append("=" * 80)
summary_lines.append("MPJPE ANALYSIS SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append(f"Total records: {len(df_normal)}")
summary_lines.append("")
summary_lines.append("--- Overall Statistics (All People) ---")
summary_lines.append(f"Mean MPJPE: {overall_mean:.2f} mm")
summary_lines.append(f"Median MPJPE: {overall_median:.2f} mm")
summary_lines.append(f"Std Dev: {overall_std:.2f} mm")
summary_lines.append(f"Min: {overall_min:.2f} mm")
summary_lines.append(f"Max: {overall_max:.2f} mm")
summary_lines.append("")
summary_lines.append("--- Best Person (Lowest Mean MPJPE) ---")
summary_lines.append(f"Person: {best_person}")
summary_lines.append(f"Mean MPJPE: {best_person_mean:.2f} mm")
summary_lines.append(f"Median MPJPE: {best_person_median:.2f} mm")
summary_lines.append(f"Std Dev: {best_person_std:.2f} mm")
summary_lines.append(f"Number of trials: {best_person_count}")
summary_lines.append("")
summary_lines.append("--- Per-Person Statistics ---")
summary_lines.append(person_stats.to_string())
summary_lines.append("")
summary_lines.append("--- Per-Song Statistics ---")
summary_lines.append(song_stats.to_string())
summary_lines.append("")
summary_lines.append("--- Per-Difficulty Statistics ---")
summary_lines.append(diff_stats.to_string())
summary_lines.append("")

with open('plots/mpjpe_analysis_summary.txt', 'w') as f:
    f.write("\n".join(summary_lines))

print("\n✓ Summary saved to plots/mpjpe_analysis_summary.txt")
print("\nDone!")
