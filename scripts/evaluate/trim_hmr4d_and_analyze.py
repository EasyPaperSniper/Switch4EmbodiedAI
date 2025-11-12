#!/usr/bin/env python3
"""
Trim hmr4d_results.pt files based on alignment_results.csv and analyze joint angle metrics.

This script:
1. Finds all alignment_results.csv files
2. Trims the corresponding hmr4d_results.pt files using best_start_idx and best_end_idx
3. Saves trimmed versions as hmr4d_results_trimmed.pt
4. Computes and reports average absolute jerk and acceleration of joint angles
"""

import pandas as pd
import numpy as np
import torch
import pathlib
from glob import glob
from tqdm import tqdm


def trim_hmr4d_results(input_path, output_path, start_idx, end_idx):
    """
    Trim hmr4d_results.pt file from start_idx to end_idx.
    
    Args:
        input_path: Path to original hmr4d_results.pt
        output_path: Path to save trimmed version
        start_idx: Starting frame index
        end_idx: Ending frame index (inclusive)
    
    Returns:
        Trimmed data dictionary
    """
    # Load original data
    data = torch.load(input_path, map_location='cpu')
    
    # Create trimmed version
    trimmed_data = {}
    
    # Trim smpl_params_global
    if 'smpl_params_global' in data:
        trimmed_data['smpl_params_global'] = {}
        for key, val in data['smpl_params_global'].items():
            if hasattr(val, '__getitem__'):
                trimmed_data['smpl_params_global'][key] = val[start_idx:end_idx+1]
            else:
                trimmed_data['smpl_params_global'][key] = val
    
    # Trim smpl_params_incam
    if 'smpl_params_incam' in data:
        trimmed_data['smpl_params_incam'] = {}
        for key, val in data['smpl_params_incam'].items():
            if hasattr(val, '__getitem__'):
                trimmed_data['smpl_params_incam'][key] = val[start_idx:end_idx+1]
            else:
                trimmed_data['smpl_params_incam'][key] = val
    
    # Trim K_fullimg
    if 'K_fullimg' in data:
        trimmed_data['K_fullimg'] = data['K_fullimg'][start_idx:end_idx+1]
    
    # Trim net_outputs (if contains tensors)
    if 'net_outputs' in data:
        trimmed_data['net_outputs'] = {}
        for key, val in data['net_outputs'].items():
            if hasattr(val, '__getitem__') and hasattr(val, 'shape') and len(val.shape) > 0 and val.shape[0] == data['K_fullimg'].shape[0]:
                trimmed_data['net_outputs'][key] = val[start_idx:end_idx+1]
            else:
                trimmed_data['net_outputs'][key] = val
    
    # Save trimmed version
    torch.save(trimmed_data, output_path)
    
    return trimmed_data

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def compute_joint_angle_metrics(body_pose, fps=30):
    """
    Compute average absolute angular acceleration and jerk of joint rotations (SMPL, 21 spherical joints).
    This version correctly handles axis–angle rotations on SO(3).
    
    Args:
        body_pose: Tensor or ndarray of shape (T, 63), axis–angle representation (21 joints × 3)
        fps: Frames per second (default: 30)
    
    Returns:
        dict: Dictionary containing metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(body_pose):
        body_pose = body_pose.cpu().numpy()
    
    T, D = body_pose.shape
    assert D == 63, f"Expected 63 DoF (21 joints × 3), got {D}"
    num_joints = 21
    dt = 1.0 / fps

    # Convert to rotation matrices
    rotmats = R.from_rotvec(body_pose.reshape(-1, 3)).as_matrix()  # (T*21, 3, 3)
    rotmats = rotmats.reshape(T, num_joints, 3, 3)

    # --- Compute angular velocity ---
    ang_vel = np.zeros((T - 1, num_joints, 3))
    for j in range(num_joints):
        for t in range(T - 1):
            R_rel = rotmats[t, j].T @ rotmats[t + 1, j]
            ang_vel[t, j] = R.from_matrix(R_rel).as_rotvec() / dt

    # --- Compute angular acceleration and jerk ---
    ang_acc = np.diff(ang_vel, axis=0) * fps          # (T-2, J, 3)
    ang_jerk = np.diff(ang_acc, axis=0) * fps         # (T-3, J, 3)

    # --- Compute magnitudes and statistics ---
    ang_acc_mag = np.linalg.norm(ang_acc, axis=-1)    # (T-2, J)
    ang_jerk_mag = np.linalg.norm(ang_jerk, axis=-1)  # (T-3, J)

    avg_abs_acc = np.mean(ang_acc_mag)
    avg_abs_jerk = np.mean(ang_jerk_mag)

    per_joint_acc = np.mean(ang_acc_mag, axis=0)      # (J,)
    per_joint_jerk = np.mean(ang_jerk_mag, axis=0)    # (J,)

    return {
        'avg_abs_acceleration': avg_abs_acc,
        'avg_abs_jerk': avg_abs_jerk,
        'per_joint_acceleration': per_joint_acc,
        'per_joint_jerk': per_joint_jerk,
        'num_frames': T
    }

def process_all_csv_files(base_dir):
    """
    Process all alignment_results.csv files found in base directory.
    
    Args:
        base_dir: Base directory to search for CSV files
    
    Returns:
        DataFrame with all results
    """
    csv_files = glob(str(base_dir / "**" / "alignment_results.csv"), recursive=True)
    
    print(f"\nFound {len(csv_files)} alignment_results.csv files")
    print(f"Searching in: {base_dir}\n")
    
    results = []
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            csv_path = pathlib.Path(csv_file)
            
            # Load CSV
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'best_start_idx' not in df.columns or 'best_end_idx' not in df.columns or 'gvhmr_1' not in df.columns:
                # Clean column names
                df.columns = df.columns.str.replace(r'\+AF8-', '_', regex=True)
                df.columns = df.columns.str.replace(r'\+AC0-', '-', regex=True)
                df.columns = df.columns.str.replace(r'\+AD0-', '=', regex=True)

                # Clean cell contents
                df = df.replace(r'\+AF8-', '_', regex=True)
                df = df.replace(r'\+AC0-', '-', regex=True)
                df = df.replace(r'\+AD0-', '=', regex=True)

                # Save cleaned version to the same file
                df.to_csv(csv_file, index=False)

                # Re-check for required columns
                if 'best_start_idx' not in df.columns or 'best_end_idx' not in df.columns or 'gvhmr_1' not in df.columns:
                    print(f"Warning: Skipping {csv_file} - missing required columns")
                    breakpoint()
                    continue
            
            # Process each row in CSV
            for idx, row in df.iterrows():
                gvhmr_1_path = pathlib.Path(row['gvhmr_1'])
                start_idx = int(row['best_start_idx'])
                end_idx = int(row['best_end_idx'])
                
                # Check if input file exists
                if not gvhmr_1_path.exists():
                    print(f"Warning: File not found: {gvhmr_1_path}")
                    continue
                
                # Define output path
                output_path = gvhmr_1_path.parent / 'hmr4d_results_trimmed.pt'
                
                # Trim the file
                trimmed_data = trim_hmr4d_results(gvhmr_1_path, output_path, start_idx, end_idx)
                
                # Compute metrics on trimmed data
                body_pose = trimmed_data['smpl_params_global']['body_pose']
                fps = row.get('target_fps', 30)
                metrics = compute_joint_angle_metrics(body_pose, fps=fps)
                
                # Get directory information
                recording_dir = csv_path.parent.name
                song_dir = csv_path.parent.parent.name
                person_dir = csv_path.parent.parent.parent.name
                
                # Store results
                results.append({
                    'person': person_dir,
                    'song': song_dir,
                    'condition': recording_dir,
                    'csv_file': str(csv_file),
                    'gvhmr_1': str(gvhmr_1_path),
                    'trimmed_file': str(output_path),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_frames_original': row.get('frames_gvhmr_1', None),
                    'num_frames_trimmed': metrics['num_frames'],
                    'avg_abs_acceleration': metrics['avg_abs_acceleration'],
                    'avg_abs_jerk': metrics['avg_abs_jerk'],
                    'target_fps': fps
                })
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(results)


def main():
    # Configuration
    base_dir = pathlib.Path("/home/jkim3662/Videos/Switch4EAI/HumanRecordings_GVHMR/raw")
    
    # Process all files
    print("=" * 80)
    print("TRIMMING HMR4D FILES AND COMPUTING JOINT ANGLE METRICS")
    print("=" * 80)
    
    results_df = process_all_csv_files(base_dir)
    
    # Save results to CSV
    output_csv = pathlib.Path("plots/hmr4d_trimming_analysis.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal files processed: {len(results_df)}")
    print(f"\nAverage Absolute Joint Angle Acceleration: {results_df['avg_abs_acceleration'].mean():.6f} ± {results_df['avg_abs_acceleration'].std():.6f}")
    print(f"Average Absolute Joint Angle Jerk: {results_df['avg_abs_jerk'].mean():.6f} ± {results_df['avg_abs_jerk'].std():.6f}")
    
    # Group by person
    print("\n" + "-" * 80)
    print("BY PERSON:")
    print("-" * 80)
    person_stats = results_df.groupby('person').agg({
        'avg_abs_acceleration': ['mean', 'std', 'count'],
        'avg_abs_jerk': ['mean', 'std']
    }).round(6)
    print(person_stats)
    
    # Group by song
    print("\n" + "-" * 80)
    print("BY SONG:")
    print("-" * 80)
    song_stats = results_df.groupby('song').agg({
        'avg_abs_acceleration': ['mean', 'std', 'count'],
        'avg_abs_jerk': ['mean', 'std']
    }).round(6)
    print(song_stats)
    
    # Create detailed report
    report_path = pathlib.Path("plots/hmr4d_joint_angle_metrics_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("JOINT ANGLE METRICS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total files processed: {len(results_df)}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Absolute Joint Angle Acceleration: {results_df['avg_abs_acceleration'].mean():.6f} ± {results_df['avg_abs_acceleration'].std():.6f}\n")
        f.write(f"  Min: {results_df['avg_abs_acceleration'].min():.6f}\n")
        f.write(f"  Max: {results_df['avg_abs_acceleration'].max():.6f}\n")
        f.write(f"  Median: {results_df['avg_abs_acceleration'].median():.6f}\n\n")
        
        f.write(f"Average Absolute Joint Angle Jerk: {results_df['avg_abs_jerk'].mean():.6f} ± {results_df['avg_abs_jerk'].std():.6f}\n")
        f.write(f"  Min: {results_df['avg_abs_jerk'].min():.6f}\n")
        f.write(f"  Max: {results_df['avg_abs_jerk'].max():.6f}\n")
        f.write(f"  Median: {results_df['avg_abs_jerk'].median():.6f}\n\n")
        
        f.write("\nBY PERSON:\n")
        f.write("-" * 80 + "\n")
        f.write(person_stats.to_string() + "\n\n")
        
        f.write("\nBY SONG:\n")
        f.write("-" * 80 + "\n")
        f.write(song_stats.to_string() + "\n\n")
        
        f.write("\nDETAILED RESULTS:\n")
        f.write("-" * 80 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"\n{row['person']} - {row['song']} - {row['condition']}\n")
            f.write(f"  Frames: {row['num_frames_original']} -> {row['num_frames_trimmed']} (trimmed)\n")
            f.write(f"  Indices: [{row['start_idx']}:{row['end_idx']}]\n")
            f.write(f"  Avg Abs Acceleration: {row['avg_abs_acceleration']:.6f}\n")
            f.write(f"  Avg Abs Jerk: {row['avg_abs_jerk']:.6f}\n")
            f.write(f"  Trimmed file: {row['trimmed_file']}\n")
    
    print(f"\nDetailed report saved to: {report_path}")
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
