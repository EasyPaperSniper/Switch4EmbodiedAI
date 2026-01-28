#!/usr/bin/env python3
"""
Compare two GMR pickle files: recorded robot trajectory vs reference motion.

Computes forward kinematics, MPJPE, and joint smoothness metrics.
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np

# Fix numpy._core compatibility for older pickle files
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

# Add GMR to path
REPO_ROOT = Path(__file__).resolve().parents[2]
GMR_ROOT = REPO_ROOT / "third_party" / "GMR"
if GMR_ROOT.exists():
    sys.path.insert(0, str(GMR_ROOT))

# Mock modules we don't need (we only use kinematics, not full GMR)
from types import ModuleType

class MockModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__file__ = f'<mock {name}>'
    
    def __getattr__(self, name):
        submod = MockModule(f'{self.__name__}.{name}')
        setattr(self, name, submod)
        return submod
    
    def __call__(self, *args, **kwargs):
        return self

sys.modules['mink'] = MockModule('mink')
sys.modules['mujoco'] = MockModule('mujoco')
sys.modules['mujoco.viewer'] = MockModule('mujoco.viewer')
sys.modules['imageio'] = MockModule('imageio')
sys.modules['loop_rate_limiters'] = MockModule('loop_rate_limiters')

from general_motion_retargeting.kinematics_model import KinematicsModel

# Add parent directory to path
import sys
import pathlib
HERE = pathlib.Path(__file__).parent
sys.path.append(str(HERE / ".." / ".."))

# Recording GMR files
from scripts.evaluate.data.gmt_sim_paths import gmr_paths, gmr_padded_paths
# from scripts.evaluate.data.twist_sim_paths import gmr_paths, gmr_padded_paths
# from scripts.evaluate.data.any2track_sim_paths import gmr_paths, gmr_padded_paths

# from scripts.evaluate.data.gmt_paths import gmr_paths, gmr_padded_paths
# from scripts.evaluate.data.twist_paths import gmr_paths, gmr_padded_paths
# from scripts.evaluate.data.any2track_paths import gmr_paths, gmr_padded_paths

# from scripts.evaluate.data.human_paths import gmr_paths, gmr_padded_paths

# Reference GMR 
REFERENCE_GMR_MAP_OFFLINE = {
    "Old_Town_Road": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/cut_mirrored/Old_Town_Road_cut/Old_Town_Road_cut_poses.pkl",
    "Heart_Of_Glass": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/cut_mirrored/Heart_Of_Glass_cut/Heart_Of_Glass_cut_poses.pkl",
    "Unstoppable": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/cut_mirrored/Unstoppable_cut/Unstoppable_cut_poses.pkl",
    "Padam_Padam": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/cut_mirrored/Padam_Padam_cut/Padam_Padam_cut_poses.pkl",
    "Pink_Venom": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/cut_mirrored/Pink_Venom_cut/Pink_Venom_cut_poses.pkl",
}
REFERENCE_GMR_MAP_ONLINE = {
    "Old_Town_Road": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Old_Town_Road/Old_Town_Road_Online_Reference.pkl",
    "Heart_Of_Glass": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Heart_Of_Glass/Heart_Of_Glass_Online_Reference.pkl",
    "Unstoppable": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Unstoppable/Unstoppable_Online_Reference.pkl",
    "Padam_Padam": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Padam_Padam/Padam_Padam_Online_Reference.pkl",
    "Pink_Venom": "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Pink_Venom/Pink_Venom_Online_Reference.pkl",
}

## Below 3 lines are debug options to compare between offline and online references
# gmr_paths = [path for path in REFERENCE_GMR_MAP_OFFLINE.values()]
# gmr_paths = [path for path in REFERENCE_GMR_MAP_ONLINE.values()]
# gmr_padded_paths = gmr_paths

# Helper function to get song name from path
def get_song_name(path):
    """Extract song name from GMR file path."""
    filename = Path(path).stem.replace("_poses", "")
    for song in REFERENCE_GMR_MAP_OFFLINE.keys():
        if filename.startswith(song):
            return song
    return None

def get_is_online(path):
    """Determine if the recording is online or offline based on filename."""
    filename = Path(path).stem
    if "online" in filename.lower():
        return True
    return False

def get_reference_map(path):
    """Get appropriate reference map based on online/offline status."""
    # For the paper, we compare against offline references only, as this measures how much the recorded motion deviates from the ideal motion.
    return REFERENCE_GMR_MAP_OFFLINE
    if get_is_online(path):
        return REFERENCE_GMR_MAP_ONLINE
    else:
        return REFERENCE_GMR_MAP_OFFLINE

# Generate corresponding reference paths
reference_gmr_paths = [get_reference_map(p)[get_song_name(p)] for p in gmr_paths]

# Robot model XML path
ROBOT_XML = REPO_ROOT / "third_party" / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"


def load_gmr_data(gmr_path):
    """Load GMR pickle file.
    
    Returns:
        dict with keys: fps, root_pos, root_rot, dof_pos, local_body_pos (optional), link_body_list (optional)
    """
    with open(gmr_path, 'rb') as f:
        data = pickle.load(f)
    return data



def align_to_zero(data):
    """Align trajectory so first frame has zero position and identity orientation.
    
    Returns:
        aligned data dict
    """
    root_pos = data['root_pos'].copy()
    root_rot = data['root_rot'].copy()
    dof_pos = data['dof_pos'].copy()
    
    # # Subtract first frame position
    # root_pos_offset = root_pos[0].copy()
    # root_pos -= root_pos_offset
    
    # # Apply inverse of first frame rotation
    # root_rot_first = R.from_quat(root_rot[0])
    # root_rot_first_inv = root_rot_first.inv()
    
    # for i in range(len(root_rot)):
    #     rot = R.from_quat(root_rot[i])
    #     rot_aligned = root_rot_first_inv * rot
    #     root_rot[i] = rot_aligned.as_quat()
    
    n_frames = dof_pos.shape[0]
    root_pos = np.zeros((n_frames, 3))  # Set root position to zeros
    root_pos[:, 2] = 1.0  # Set a constant height for the root (e.g., z=1.0)
    root_rot_wxyz = np.zeros((n_frames, 4))
    root_rot_wxyz[:, 0] = 1.0  # Set w component to 1 (no rotation)
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]  # Convert wxyz to xyzw if needed
    return {
        'fps': data['fps'],
        'root_pos': root_pos,
        'root_rot': root_rot_xyzw,
        'dof_pos': dof_pos,
    }


def compute_forward_kinematics(xml_file, data, device='cpu', batch_size=256):
    """Compute forward kinematics using GMR KinematicsModel.
    
    Args:
        xml_file: Path to robot MJCF file
        data: dict with root_pos, root_rot, dof_pos
        device: torch device
        batch_size: number of frames to process at once
    
    Returns:
        body_positions: (T, num_bodies, 3) numpy array
        body_names: list of body names
    """
    device = torch.device(device)
    kinematics_model = KinematicsModel(str(xml_file), device=device)
    
    root_pos = data['root_pos']
    root_rot = data['root_rot']
    dof_pos = data['dof_pos']
    
    T = root_pos.shape[0]
    body_positions_list = []
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, T, batch_size):
        end_idx = min(start_idx + batch_size, T)
        
        root_pos_batch = torch.from_numpy(root_pos[start_idx:end_idx]).to(device=device, dtype=torch.float32)
        root_rot_batch = torch.from_numpy(root_rot[start_idx:end_idx]).to(device=device, dtype=torch.float32)
        dof_pos_batch = torch.from_numpy(dof_pos[start_idx:end_idx]).to(device=device, dtype=torch.float32)
        
        body_pos_batch, body_rot_batch = kinematics_model.forward_kinematics(
            root_pos_batch,
            root_rot_batch,
            dof_pos_batch,
        )
        
        body_positions_list.append(body_pos_batch.detach().cpu().numpy())
    
    body_positions = np.concatenate(body_positions_list, axis=0)
    body_names = kinematics_model.body_names
    
    return body_positions, body_names


def compute_mpjpe(recorded_pos, reference_pos):
    """Compute Mean Per Joint Position Error.
    
    Args:
        recorded_pos: (T, num_joints, 3)
        reference_pos: (T, num_joints, 3)
    
    Returns:
        mpjpe: mean per joint position error (meters)
        per_joint_error: (num_joints,) array of mean error per joint
        per_frame_error: (T,) array of mean error per frame
    """
    # Euclidean distance per joint per frame
    diff = recorded_pos - reference_pos
    distances = np.linalg.norm(diff, axis=2)  # (T, num_joints)
    
    # Mean across all frames and joints
    mpjpe = np.mean(distances)
    
    # # Mean per joint (across time)
    # per_joint_error = np.mean(distances, axis=0)
    
    # # Mean per frame (across joints)
    # per_frame_error = np.mean(distances, axis=1)
    
    return mpjpe


def compute_joint_smoothness(dof_pos, fps):
    """Compute joint smoothness using jerk and velocity discontinuities.
    
    Args:
        dof_pos: (T, num_dof) joint positions
        fps: framerate (samples per second)
    
    Returns:
        smoothness: mean jerk magnitude (rad/s^3)
        velocity_discontinuity: mean velocity jump magnitude (rad/s)
    """
    dt = 1.0 / fps
    
    # 1st derivative: velocity
    velocity = np.diff(dof_pos, axis=0) / dt  # (T-1, num_dof)
    mean_velocity = np.mean(np.linalg.norm(velocity, axis=1))
    
    # 2nd derivative: acceleration
    acceleration = np.diff(velocity, axis=0) / dt  # (T-2, num_dof)
    mean_acceleration = np.mean(np.linalg.norm(acceleration, axis=1))
    
    # 3rd derivative: jerk
    jerk = np.diff(acceleration, axis=0) / dt  # (T-3, num_dof)
    
    # Smoothness = mean jerk magnitude
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    smoothness = np.mean(jerk_magnitude)
    
    # Velocity discontinuity = magnitude of velocity jumps
    vel_jump = np.diff(velocity, axis=0)
    velocity_discontinuity = np.mean(np.linalg.norm(vel_jump, axis=1))
    
    
    return smoothness, velocity_discontinuity, mean_velocity, mean_acceleration


def find_optimal_time_alignment_mpjpe(recorded_body_pos, reference_body_pos, max_shift=None):
    """Find optimal time alignment by minimizing MPJPE.
    
    This is a faster approximation that works directly on joint angles instead of 3D positions.
    
    Args:
        recorded_body_pos: (T1, num_bodies, 3) recorded body positions
        reference_body_pos: (T2, num_bodies, 3) reference body positions (T2 <= T1)
        max_shift: maximum shift to search (default: T1 - T2 + 1)
    
    Returns:
        recorded_body_pos_aligned: (T, num_bodies, 3) aligned recorded body positions
        reference_body_pos_aligned: (T, num_bodies, 3) reference body positions
        recorded_aligned_indices: indices used for alignment
        referece_aligned_indices: indices used for alignment
        best_error: best MPJPE achieved
    """
    T1 = recorded_body_pos.shape[0]
    T2 = reference_body_pos.shape[0]

    if T1 >= T2:
        T = T2
        if max_shift is None:
            max_shift = T1 - T2 + 1
        else:
            max_shift = min(max_shift, T1 - T2 + 1)
        
        best_error = float('inf')
        best_start_idx = 0
        
        for start_idx in tqdm(range(max_shift), desc="Finding optimal alignment", unit="shift", leave=False):
            recorded_shifted = recorded_body_pos[start_idx:start_idx+T2]
            error = compute_mpjpe(recorded_shifted, reference_body_pos)

            if error < best_error:
                best_error = error
                best_start_idx = start_idx
        
        recorded_aligned_indices = range(best_start_idx, best_start_idx + T)
        reference_aligned_indices = range(T)    
        return recorded_body_pos[best_start_idx:best_start_idx+T], reference_body_pos, recorded_aligned_indices, reference_aligned_indices, best_error

    else: # T1 < T2
        T = T1
        if max_shift is None:
            max_shift = T2 - T1 + 1
        else:
            max_shift = min(max_shift, T2 - T1 + 1)
        
        best_error = float('inf')
        best_start_idx = 0
        
        for start_idx in tqdm(range(max_shift), desc="Finding optimal alignment", unit="shift", leave=False):
            reference_shifted = reference_body_pos[start_idx:start_idx+T1]
            error = compute_mpjpe(recorded_body_pos, reference_shifted)

            if error < best_error:
                best_error = error
                best_start_idx = start_idx

        recorded_aligned_indices = range(T)
        reference_aligned_indices = range(best_start_idx, best_start_idx + T)
        return recorded_body_pos, reference_body_pos[best_start_idx:best_start_idx+T], recorded_aligned_indices, reference_aligned_indices, best_error

    raise RuntimeError("Should not reach here")


def save_results(recorded_path, results):
    """Save results as JSON next to the recorded GMR file."""
    output_path = Path(recorded_path).with_suffix('.comparison_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)


for i, recorded_gmr_path in enumerate(tqdm(gmr_paths, desc="Processing recordings")):
    reference_gmr_path = reference_gmr_paths[i]
    
    try:
        # Load data
        recorded_data = load_gmr_data(recorded_gmr_path)
        reference_data = load_gmr_data(reference_gmr_path)
        
        # Align trajectories to zero first (before time alignment)
        recorded_data = align_to_zero(recorded_data)
        reference_data = align_to_zero(reference_data)
        
        # Compute forward kinematics
        recorded_body_pos, body_names = compute_forward_kinematics(ROBOT_XML, recorded_data)
        reference_body_pos, _ = compute_forward_kinematics(ROBOT_XML, reference_data)

        
        # Find optimal time alignment on MPJPE
        recorded_body_pos_aligned, reference_body_pos_aligned, recorded_aligned_indices, reference_aligned_indices, _ = find_optimal_time_alignment_mpjpe(
            recorded_body_pos,
            reference_body_pos
        )
        
        n_frames = recorded_body_pos_aligned.shape[0]

        UPPERBODY_INDICES = [15, 16, 17, 18, 19, 20, 21, 22]  # Arms
        # Compute metrics
        mpjpe = compute_mpjpe(recorded_body_pos_aligned, reference_body_pos_aligned)
        recorded_smoothness, recorded_velocity_discontinuity, recorded_mean_velocity, recorded_mean_acceleration = compute_joint_smoothness(
            recorded_data['dof_pos'][recorded_aligned_indices], recorded_data['fps']
        )
        reference_smoothness, reference_velocity_discontinuity, reference_mean_velocity, reference_mean_acceleration = compute_joint_smoothness(
            reference_data['dof_pos'][reference_aligned_indices], reference_data['fps']
        )

        if True:
            recorded_padded_gmr_path = gmr_padded_paths[i]
            recorded_padded_data = load_gmr_data(recorded_padded_gmr_path)
            recorded_padded_data = align_to_zero(recorded_padded_data)
            recorded_padded_body_pos, _ = compute_forward_kinematics(ROBOT_XML, recorded_padded_data)
            # Find optimal time alignment on MPJPE
            recorded_padded_body_pos_aligned, reference_padded_body_pos_aligned, recorded_aligned_indices, reference_aligned_indices, _ = find_optimal_time_alignment_mpjpe(
                recorded_padded_body_pos,
                reference_body_pos
            )

            n_frames_padded = recorded_padded_body_pos_aligned.shape[0]
            # Compute metrics
            mpjpe_padded = compute_mpjpe(recorded_padded_body_pos_aligned, reference_padded_body_pos_aligned)

        
        # Save results
        results = {
            'recorded_path': recorded_gmr_path,
            'recorded_padded_path': recorded_padded_gmr_path,
            'reference_path': reference_gmr_path,
            'n_frames': n_frames,
            'fps': recorded_data['fps'],
            'mpjpe_mm': float(mpjpe * 1000),
            'mpjpe_padded_mm': float(mpjpe_padded * 1000),
            # 'body_names': body_names,
            'recorded_smoothness': float(recorded_smoothness),
            # 'reference_smoothness': float(reference_smoothness),
            # 'smoothness_ratio': float(recorded_smoothness / reference_smoothness),
            # 'recorded_velocity_discontinuity': float(recorded_velocity_discontinuity),
            # 'reference_velocity_discontinuity': float(reference_velocity_discontinuity),
            # 'recorded_mean_velocity': float(recorded_mean_velocity),
            # 'reference_mean_velocity': float(reference_mean_velocity),
            'recorded_mean_acceleration': float(recorded_mean_acceleration),
            # 'reference_mean_acceleration': float(reference_mean_acceleration),
        }
        save_results(recorded_gmr_path, results)
        
        # Print summary
        tqdm.write(
            f"{Path(recorded_gmr_path).stem} | "
            f"Frames: {n_frames} | "
            f"MPJPE: {mpjpe*1000:.1f} mm | "
            f"Frames_padded: {n_frames_padded} | "
            f"MPJPE_padded: {mpjpe_padded*1000:.1f} mm | "
            f"Smooth: {recorded_smoothness:.2f} rad/s³ | "
            # f"VelDisc: {recorded_velocity_discontinuity:.2f} rad/s | "
            f"MeanAcc: {recorded_mean_acceleration:.2f} rad/s² | "
            # f"MeanVel: {recorded_mean_velocity:.2f} rad/s"
        )
        
    except Exception as e:
        tqdm.write(f"ERROR processing {Path(recorded_gmr_path).name}: {e}")
        continue

print("\n✓ All comparisons complete!")

####################
# Generate summary #
####################

def generate_summary(recorded_paths):
    """Generate a summary of all comparison results."""
    from collections import defaultdict
    from datetime import datetime
    
    # Collect all result files
    results_by_song = defaultdict(lambda: {"offline": [], "online": []})
    
    for recorded_path in recorded_paths:
        result_file = Path(recorded_path).with_suffix('.comparison_results.json')
        if not result_file.exists():
            continue
            
        with open(result_file) as f:
            data = json.load(f)
        
        # Extract song name and mode
        song = get_song_name(recorded_path)
        mode = "online" if get_is_online(recorded_path) else "offline"
        results_by_song[song][mode].append(data)

    # Generate summary text
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("GMR TRAJECTORY COMPARISON RESULTS")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Analysis Date: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    summary_lines.append(f"Total Comparisons: {len(recorded_paths)}")
    summary_lines.append("")
    
    # Per-song results
    # for song in sorted(results_by_song.keys()):
    for song in results_by_song.keys():
        summary_lines.append(f"{song}:")
        for mode in ["online", "offline"]:
            trials = results_by_song[song][mode]
            # Only take trials that lasted more than 300 frames (10 seconds at 30 FPS)
            trials = [t for t in trials if t["n_frames"] >= 300]
            if trials:
                mpjpes = [t["mpjpe_mm"] for t in trials]
                mpjpes_padded = [t["mpjpe_padded_mm"] for t in trials]
                smooth = [t["recorded_smoothness"] for t in trials]
                mean_acc = [t["recorded_mean_acceleration"] for t in trials]
                summary_lines.append(
                    f"  {mode.capitalize()} (n={len(trials)}): "
                    f"MPJPE={sum(mpjpes)/len(mpjpes):.1f} mm | "
                    f"MPJPE_padded={sum(mpjpes_padded)/len(mpjpes_padded):.1f} mm | "
                    f"Smooth={sum(smooth)/len(smooth):.2f} rad/s³ | "
                    f"MeanAcc={sum(mean_acc)/len(mean_acc):.2f} rad/s²"
                )
    
    # Overall statistics
    all_offline = []
    all_online = []
    for song_data in results_by_song.values():
        all_online.extend(song_data["online"])
        all_offline.extend(song_data["offline"])
    
    summary_lines.append("")
    if all_online:
        mpjpes = [t["mpjpe_mm"] for t in all_online if t["n_frames"] >= 300]
        mpjpes_padded = [t["mpjpe_padded_mm"] for t in all_online if t["n_frames"] >= 300]
        smooth = [t["recorded_smoothness"] for t in all_online if t["n_frames"] >= 300]
        mean_acc = [t["recorded_mean_acceleration"] for t in all_online if t["n_frames"] >= 300]
        summary_lines.append(f"Overall Online (n={len(mpjpes)}): MPJPE = {sum(mpjpes)/len(mpjpes):.1f} mm, MPJPE_padded = {sum(mpjpes_padded)/len(mpjpes_padded):.1f} mm, Smooth = {sum(smooth)/len(smooth):.2f} rad/s³, MeanAcc = {sum(mean_acc)/len(mean_acc):.2f} rad/s²")
    if all_offline:
        mpjpes = [t["mpjpe_mm"] for t in all_offline if t["n_frames"] >= 300]
        mpjpes_padded = [t["mpjpe_padded_mm"] for t in all_offline if t["n_frames"] >= 300]
        smooth = [t["recorded_smoothness"] for t in all_offline if t["n_frames"] >= 300]
        mean_acc = [t["recorded_mean_acceleration"] for t in all_offline if t["n_frames"] >= 300]
        summary_lines.append(f"Overall Offline (n={len(mpjpes)}): MPJPE = {sum(mpjpes)/len(mpjpes):.1f} mm, MPJPE_padded = {sum(mpjpes_padded)/len(mpjpes_padded):.1f} mm, Smooth = {sum(smooth)/len(smooth):.2f} rad/s³, MeanAcc = {sum(mean_acc)/len(mean_acc):.2f} rad/s²")

    summary_lines.append("=" * 80)
    
    # Write to file
    output_dir = REPO_ROOT / "plots" / "gmr"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comparison_summary.txt"
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Summary saved to: {output_file}")
    print('\n'.join(summary_lines))

generate_summary(gmr_paths)
