import pathlib
import torch
import numpy as np
import argparse
import os
import time
import csv
import subprocess
from tqdm import tqdm
HERE = pathlib.Path(__file__).parent
os.sys.path.append(str(HERE / ".." / ".."))

# pip install smplx
from scripts.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

import numpy as np

# ---------- Utilities ----------
def root_center(poses, pelvis_idxs=[1, 2]):
    """
    poses: (T, J, 3) array
    Subtract the root (pelvis) so poses become root-relative.
    """
    pred_pelvis = poses[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    return poses - pred_pelvis                                     # (T,J,3)


def compute_velocity(data, fps=30.0):
    """
    Compute velocity using central finite differences.
    
    Args:
        data: torch.Tensor of shape (T, ...) where T is time dimension
        fps: frames per second for time normalization
        
    Returns:
        velocity: torch.Tensor of shape (T, ...) in units/second
    """
    dt = 1.0 / fps
    velocity = torch.zeros_like(data)
    
    # Forward difference for first frame
    velocity[0] = (data[1] - data[0]) / dt
    
    # Central difference for middle frames
    velocity[1:-1] = (data[2:] - data[:-2]) / (2 * dt)
    
    # Backward difference for last frame
    velocity[-1] = (data[-1] - data[-2]) / dt
    
    return velocity


def compute_acceleration(data, fps=30.0):
    """
    Compute acceleration using central finite differences on velocity.
    
    Args:
        data: torch.Tensor of shape (T, ...) where T is time dimension
        fps: frames per second for time normalization
        
    Returns:
        acceleration: torch.Tensor of shape (T, ...) in units/second^2
    """
    velocity = compute_velocity(data, fps)
    acceleration = compute_velocity(velocity, fps)  # velocity of velocity
    return acceleration


def compute_kinematic_metrics(pred_data, target_data, fps=30.0, metric_name="position"):
    """
    Compute velocity and acceleration errors between predicted and target sequences.
    
    Args:
        pred_data: torch.Tensor of shape (T, J, 3) or (T, num_params)
        target_data: torch.Tensor of shape (T, J, 3) or (T, num_params)
        fps: frames per second
        metric_name: name for logging (e.g., "position", "joint_angle")
        
    Returns:
        dict with velocity and acceleration metrics
    """
    # Compute velocities
    pred_vel = compute_velocity(pred_data, fps)
    target_vel = compute_velocity(target_data, fps)
    
    # Compute accelerations
    pred_acc = compute_acceleration(pred_data, fps)
    target_acc = compute_acceleration(target_data, fps)
    
    # Compute errors (L2 norm across last dimensions)
    vel_error = torch.sqrt(((pred_vel - target_vel) ** 2).sum(dim=-1))  # (T, J) or (T,)
    acc_error = torch.sqrt(((pred_acc - target_acc) ** 2).sum(dim=-1))  # (T, J) or (T,)
    
    # If we have joint dimension, average over joints
    if vel_error.ndim > 1:
        vel_error = vel_error.mean(dim=-1)  # (T,)
        acc_error = acc_error.mean(dim=-1)  # (T,)
    
    # Compute mean and std
    metrics = {
        f'{metric_name}_velocity_error_mean': vel_error.mean().item(),
        f'{metric_name}_velocity_error_std': vel_error.std().item(),
        f'{metric_name}_acceleration_error_mean': acc_error.mean().item(),
        f'{metric_name}_acceleration_error_std': acc_error.std().item(),
    }
    
    return metrics

# ---------- Per-Frame Procrustes (similarity) alignment via Kabsch + scale ----------
def compute_jpe(S1, S2):
    # S1, S2: (frames, num_joints, 3)
    result = torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1)
    return result.cpu().numpy() if result.is_cuda else result.numpy()  # (frames,)


def compute_perjoint_jpe(S1, S2):
    # S1, S2: (frames, num_joints, 3)
    result = torch.sqrt(((S1 - S2) ** 2).sum(dim=-1))
    return result.cpu().numpy() if result.is_cuda else result.numpy()  # (frames, num_joints)


def batch_align_by_pelvis(data_list, pelvis_idxs=[1, 2]):
    """
    Assumes data is given as [pred_j3d, target_j3d, pred_verts, target_verts].
    Each data is in shape of (frames, num_points, 3)
    Pelvis is notated as one / two joints indices, defaults to using [1,2] using two hip joints.
    Align all data to the corresponding pelvis location.
    """

    pred_j3d, target_j3d, pred_verts, target_verts = data_list  # each: (frames, num_points, 3)

    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()    # (frames, 1, 3)
    target_pelvis = target_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()  # (frames, 1, 3)

    # Align to the pelvis (translation only)
    pred_j3d = pred_j3d - pred_pelvis          # (frames, num_points, 3)
    target_j3d = target_j3d - target_pelvis    # (frames, num_points, 3)
    if pred_verts is not None:
        pred_verts = pred_verts - pred_pelvis      # (frames, num_points, 3)
    if target_verts is not None:
        target_verts = target_verts - target_pelvis  # (frames, num_points, 3)

    return (pred_j3d, target_j3d, pred_verts, target_verts)

def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a *similarity transform* (s, R, t) that best aligns a batch of 3D point sets S1 to S2
    in the least-squares sense — the classic *Orthogonal Procrustes* problem.

    Mathematically, for each batch b:
        Find scale s_b ∈ ℝ, rotation R_b ∈ SO(3), and translation t_b ∈ ℝ³
        that minimize:
            || s_b * R_b * S1_b + t_b - S2_b ||_F²

        The closed-form solution is:
            μ₁ = mean(S1_b, axis=-1)
            μ₂ = mean(S2_b, axis=-1)
            X₁ = S1_b - μ₁
            X₂ = S2_b - μ₂
            K  = X₁ X₂ᵀ
            [U, Σ, V] = svd(K)
            Z = diag(1, 1, det(UVᵀ))
            R_b = V Z Uᵀ
            s_b = trace(R_b K) / ||X₁||_F²
            t_b = μ₂ - s_b * R_b * μ₁
            S1_hat_b = s_b * R_b * S1_b + t_b

    Args:
        S1: torch.Tensor
            Predicted 3D points, shape (B, N, 3) or (B, 3, N)
        S2: torch.Tensor
            Target 3D points (same correspondence), shape (B, N, 3) or (B, 3, N)

            B = batch size (e.g., number of frames)
            N = number of points per sample
            3 = coordinate dimension (x, y, z)

    Returns:
        S1_hat: torch.Tensor
            Aligned version of S1 after applying the optimal similarity transform.
            Shape is the same as the input (B, N, 3) or (B, 3, N), matching S1.

    Notes:
        • Translation, rotation, and isotropic scale are all estimated per batch.
        • Internally, tensors are converted to (B, 3, N) for matrix operations.
        • Ensures det(R) = 1 (no reflections).
        • Commonly used for computing PA-MPJPE (Procrustes Aligned Mean Per Joint Position Error).
    """
    transposed = False
    if S1.shape[1] != 3 and S1.shape[1] != 2:
        S1 = S1.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
        S2 = S2.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
        transposed = True

    assert S2.shape[-1] == S1.shape[-1], "Number of points (N) must match between S1 and S2"

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)  # (B, 3, 1)
    mu2 = S2.mean(axis=-1, keepdims=True)  # (B, 3, 1)

    X1 = S1 - mu1  # (B, 3, N)
    X2 = S2 - mu2  # (B, 3, N)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)  # (B,)

    # 3. Cross-covariance.
    K = X1.bmm(X2.permute(0, 2, 1))  # (B, 3, N) x (B, N, 3) -> (B, 3, 3)

    # 4. Optimal rotation via SVD.
    U, s, V = torch.svd(K)
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0).repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))  # (B, 3, 3)

    # 5. Scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1  # (B,)

    # 6. Translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))  # (B, 3, 1)

    # 7. Apply transform.
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t  # (B, 3, N)

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)  # (B, 3, N) -> (B, N, 3)

    return S1_hat

# ---------- Main metric function ----------
def compute_perjoint_metrics(pred_j3d, target_j3d, pelvis_idxs=[1, 2], fps=30.0):
    """
    seq1, seq2: (T, J, 3), (T, J, 3) in same joint order and units (e.g., m)
    Returns:
      {
        'pa_mpjpe': float, # mean per-joint position error after global PA
        'mpjpe': float,    # mean per-joint position error only with pelvis alignment
        'position_velocity_error_mean': float, # mean velocity error (m/s)
        'position_velocity_error_std': float,
        'position_acceleration_error_mean': float, # mean acceleration error (m/s^2)
        'position_acceleration_error_std': float,
      }
    
    Note: For optimal performance, pass tensors directly to avoid repeated conversions.
    """
    assert pred_j3d.shape[0] == target_j3d.shape[0], "The number of frames in pred_j3d and target_j3d must be the same."

    # Send the values to torch tensors (only if not already tensors)
    if not torch.is_tensor(pred_j3d):
        pred_j3d = torch.tensor(pred_j3d, dtype=torch.float32)
    if not torch.is_tensor(target_j3d):
        target_j3d = torch.tensor(target_j3d, dtype=torch.float32)

    # Store original data for kinematic analysis
    pred_j3d_original = pred_j3d.clone()
    target_j3d_original = target_j3d.clone()

    # (Optional) root-center (subtract the root joint position from all joints)
    # pred_j3d = root_center(pred_j3d, pelvis_idxs=pelvis_idxs)
    # target_j3d = root_center(target_j3d, pelvis_idxs=pelvis_idxs)


    # Align by pelvis
    pred_j3d, target_j3d, _, _ = batch_align_by_pelvis(
        [pred_j3d, target_j3d, None, None], pelvis_idxs=pelvis_idxs
    )

    # Metrics
    # Per-frame Procrustes alignment of pred_j3d to target_j3d
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d).mean() # (num_frames,) -> float
    mpjpe = compute_jpe(pred_j3d, target_j3d).mean() # (num_frames,) -> float
    
    # Compute kinematic metrics (velocity and acceleration) on aligned data
    kinematic_metrics = compute_kinematic_metrics(pred_j3d, target_j3d, fps=fps, metric_name="position")
    
    perjoint_metrics = { # per-joint metrics in numpy array
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
    }
    
    # Add kinematic metrics
    perjoint_metrics.update(kinematic_metrics)
    
    return perjoint_metrics


# ---------- Pairwise PA-MPJPE cost matrix + DTW ----------

def compute_pairwise_pa_mpjpe_matrix(pred_j3d, target_j3d, pelvis_idxs=[1, 2], device=None, chunk=None, show_progress=True):
    """
    Build a cost matrix C where C[i, j] = PA-MPJPE(pred_frame_i, target_frame_j).
    pred_j3d:  (T1, J, 3) torch.Tensor
    target_j3d:(T2, J, 3) torch.Tensor
    Returns:
        C: (T1, T2) numpy array (float32), in millimeters (matches your compute_jpe unit)
    """
    assert torch.is_tensor(pred_j3d) and torch.is_tensor(target_j3d), "Inputs must be torch tensors"
    assert pred_j3d.ndim == 3 and target_j3d.ndim == 3 and pred_j3d.shape[1:] == target_j3d.shape[1:], "Shape mismatch"
    T1, J, _ = pred_j3d.shape
    T2 = target_j3d.shape[0]

    if device is None:
        device = pred_j3d.device

    # Commented out because using CPU was faster in our case.
    # if torch.cuda.is_available() and device.type == 'cpu':
    #     device = torch.device('cuda')
    #     print(f"  Using GPU ({torch.cuda.get_device_name(0)}) for acceleration")
    
    pred = pred_j3d.to(device)
    targ = target_j3d.to(device)

    # Pelvis-center each frame independently (translation only)
    pred_c, targ_c, _, _ = batch_align_by_pelvis([pred, targ, None, None], pelvis_idxs=pelvis_idxs)  # (T1,J,3), (T2,J,3)

    # Build pairwise PA-MPJPE with batched Procrustes
    C = np.zeros((T1, T2), dtype=np.float32)
    
    iterator = tqdm(range(T1), desc="Computing cost matrix", unit="frame") if show_progress else range(T1)
    
    with torch.no_grad():
        for i in iterator:
            # Compare frame i to all target frames at once (vectorized over T2)
            S1 = pred_c[i:i+1].repeat(T2, 1, 1)        # (T2, J, 3)
            S2 = targ_c                                 # (T2, J, 3)

            # If very long sequences, you can chunk along T2 to save memory
            if chunk is None:
                S1_hat = batch_compute_similarity_transform_torch(S1, S2)   # (T2, J, 3)
                costs = compute_jpe(S1_hat, S2)                              # (T2,)
                C[i, :] = costs
            else:
                k = 0
                while k < T2:
                    kk = min(k + chunk, T2)
                    S1_hat = batch_compute_similarity_transform_torch(S1[k:kk], S2[k:kk])  # (kk-k, J, 3)
                    costs = compute_jpe(S1_hat, S2[k:kk])                                  # (kk-k,)
                    C[i, k:kk] = costs
                    k = kk
    return C


def dtw_from_cost_matrix(C, band=None):
    """
    Standard DTW on a precomputed cost matrix C (T1 x T2).
    Steps allowed: (1,0), (0,1), (1,1). No slope weighting.
    band: optional Sakoe-Chiba band (int). If set, enforces |i - j| <= band.

    Returns:
        total_cost: float
        path: list of (i, j) indices, monotonic warping path from (0,0) to (T1-1, T2-1)
        D: accumulated cost matrix (T1 x T2) as numpy array
    """
    T1, T2 = C.shape
    D = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0

    # Forward pass with optional banding
    for i in range(1, T1 + 1):
        j_min = 1
        j_max = T2 + 1
        if band is not None:
            j_min = max(1, i - band)
            j_max = min(T2 + 1, i + band + 1)
        for j in range(j_min, j_max):
            d = C[i - 1, j - 1]
            D[i, j] = d + min(D[i - 1, j],     # insertion (i-1, j)
                              D[i, j - 1],     # deletion  (i, j-1)
                              D[i - 1, j - 1]) # match     (i-1, j-1)

    # Backtrack
    i, j = T1, T2
    path = [(i - 1, j - 1)]
    
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            # min finding
            candidates = np.array([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            move = candidates.argmin()
            if move == 0:    # came from (i-1, j)
                i -= 1
            elif move == 1:  # came from (i, j-1)
                j -= 1
            else:            # came from (i-1, j-1)
                i -= 1
                j -= 1
        path.append((i - 1, j - 1))
    
    path.reverse()
    return float(D[T1, T2]), path, D[1:, 1:]


def cut_video_with_ffmpeg(input_video_path, output_video_path, start_frame, end_frame, fps):
    """
    Cut a video using ffmpeg with frame-based precision.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        start_frame: Start frame number (0-indexed)
        end_frame: End frame number (exclusive)
        fps: Frames per second of the video
    
    Returns:
        True if successful, False otherwise
    """
    # Use frame-based filtering for exact frame selection
    num_frames = end_frame - start_frame
    
    cmd = [
        "ffmpeg",
        "-i", str(input_video_path),
        "-vf", f"select='between(n\\,{start_frame}\\,{end_frame-1})',setpts=PTS-STARTPTS",
        "-af", f"aselect='between(n\\,{start_frame}\\,{end_frame-1})',asetpts=PTS-STARTPTS",
        "-y", str(output_video_path),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: ffmpeg returned non-zero exit code: {result.returncode}")
        print(f"stderr: {result.stderr}")
        return False
    else:
        print(f"✓ Successfully cut video: {output_video_path}")
        return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_1",
        help="First SMPLX motion file to load. We assume the file is generated from GVHMR.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    )
    parser.add_argument(
        "--gvhmr_2",
        help="Second SMPLX motion file to load. We assume the file is generated from GVHMR.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    )
    parser.add_argument(
        "--csv_output",
        help="Path to CSV file for saving metrics (default: alignment_results.csv in same directory as gvhmr_1)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cut_videos",
        help="Cut videos based on optimal alignment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--video_names",
        help="Names of video files to cut (default: 0_input_video.mp4 1_incam.mp4)",
        type=str,
        nargs='+',
        default=['0_input_video.mp4', '1_incam.mp4'],
    )
    parser.add_argument(
        "--compute_dtw",
        help="Compute DTW alignment on the optimal cut segment (default: True)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_dtw",
        help="Skip DTW computation",
        action="store_false",
        dest="compute_dtw",
    )
    args = parser.parse_args()
    
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models" / "smplx"
    SMPLX_FOLDER = "assets/body_models"
    TRIM_SECONDS = 0  # seconds to trim from start and end of target sequence
    # Load First SMPLX trajectory
    smplx_data_1, body_model_1, smplx_output_1, actual_human_height_1 = load_gvhmr_pred_file(
        args.gvhmr_1, SMPLX_FOLDER
    )
    # align fps
    tgt_fps = 30
    smplx_data_frames_1, aligned_fps_1 = get_gvhmr_data_offline_fast(smplx_data_1, body_model_1, smplx_output_1, tgt_fps=tgt_fps)

    # Load Second SMPLX trajectory
    smplx_data_2, body_model_2, smplx_output_2, actual_human_height_2 = load_gvhmr_pred_file(
        args.gvhmr_2, SMPLX_FOLDER
    )
    # align fps
    smplx_data_frames_2, aligned_fps_2 = get_gvhmr_data_offline_fast(smplx_data_2, body_model_2, smplx_output_2, tgt_fps=tgt_fps)
    
    smpl_joints = [ # 22 joints for SMPL, 24 for SMPL+H
        'pelvis', 'left_hip', 'right_hip',
        'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle',
        'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar',
        'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        #'left_hand', 'right_hand'  # SMPL+H (22+2 joints)
    ]

    # Suppose `joints_dict` is your SMPL-X joints
    def get_smpl_joint_position_array(smplx_data_frames):
        joints_array = []
        for frame_data in smplx_data_frames:
            frame_joints = []
            for joint in smpl_joints:
                if joint in frame_data:
                    position, _ = frame_data[joint]
                    frame_joints.append(position)
                else:
                    raise ValueError(f"Joint '{joint}' not found in the data.")
            joints_array.append(np.array(frame_joints))  # Shape: (num_joints, 3)
        return np.array(joints_array)  # Shape: (num_frames, num_joints, 3)
    pred_j3d = get_smpl_joint_position_array(smplx_data_frames_1)
    target_j3d = get_smpl_joint_position_array(smplx_data_frames_2)

    # Trim first and last 2 seconds from target_j3d to avoid boundary effects
    trim_seconds = TRIM_SECONDS
    trim_frames = int(trim_seconds * tgt_fps)
    if trim_frames > 0:
        target_j3d = target_j3d[trim_frames:-trim_frames]
    print(f"Trimmed {trim_seconds} seconds ({trim_frames} frames) from start and end of target sequence.")

    T1 = pred_j3d.shape[0]
    T2 = target_j3d.shape[0]
    print(f"Frames in GVHMR 1: {T1}, Frames in GVHMR 2: {T2}")
    assert T1 >= T2, "The first sequence must be the same length or longer than the second."
    
    # Convert to torch tensors once before the loop for efficiency
    print("\nConverting to torch tensors...")
    if not torch.is_tensor(pred_j3d):
        pred_j3d = torch.tensor(pred_j3d, dtype=torch.float32)
    if not torch.is_tensor(target_j3d):
        target_j3d = torch.tensor(target_j3d, dtype=torch.float32)
    
    # Find optimal time alignment by minimizing PA-MPJPE
    best_pa_mpjpe = float('inf')
    best_mpjpe = float('inf')
    best_start_idx = 0
    
    print("\nSearching for optimal time alignment...")
    max_shift = T1 - T2 + 1  # number of possible starting positions
    
    start_time = time.time()
    iteration_times = []
    
    for start_idx in tqdm(range(max_shift), desc="Finding optimal alignment", unit="shift"):
        iter_start = time.time()
        
        pred_j3d_shifted = pred_j3d[start_idx:start_idx+T2]
        result = compute_perjoint_metrics(pred_j3d_shifted, target_j3d, pelvis_idxs=[1,2])
        pa_mpjpe = result['pa_mpjpe'].item() if torch.is_tensor(result['pa_mpjpe']) else result['pa_mpjpe']
        mpjpe = result['mpjpe'].item() if torch.is_tensor(result['mpjpe']) else result['mpjpe']
        
        if pa_mpjpe < best_pa_mpjpe:
            best_pa_mpjpe = pa_mpjpe
            best_mpjpe = mpjpe
            best_start_idx = start_idx
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
    
    total_time = time.time() - start_time
    avg_time_per_iter = np.mean(iteration_times)
    
    print(f"\nTiming Statistics:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per iteration: {avg_time_per_iter*1000:.2f} ms")
    print(f"  Total iterations: {max_shift}")
    
    print(f"\n{'='*60}")
    print(f"Optimal Time Alignment Found:")
    print(f"{'='*60}")
    print(f"Best start_idx: {best_start_idx}")
    print(f"Optimal PA-MPJPE: {best_pa_mpjpe:.6f}")
    print(f"Corresponding MPJPE: {best_mpjpe:.6f}")
    print(f"{'='*60}")
    
    # Compute and display final result with optimal alignment
    pred_j3d_optimal = pred_j3d[best_start_idx:best_start_idx+T2]
    result_optimal = compute_perjoint_metrics(pred_j3d_optimal, target_j3d, pelvis_idxs=[1,2], fps=tgt_fps)
    print("\nFinal verification:")
    print(f"Sequence-level PA-MPJPE: {result_optimal['pa_mpjpe']:.6f}")
    print(f"Sequence-level MPJPE: {result_optimal['mpjpe']:.6f}")
    print(f"Position Velocity Error: {result_optimal['position_velocity_error_mean']:.6f} ± {result_optimal['position_velocity_error_std']:.6f} m/s")
    print(f"Position Acceleration Error: {result_optimal['position_acceleration_error_mean']:.6f} ± {result_optimal['position_acceleration_error_std']:.6f} m/s²")
    
    # Also compute joint angle metrics if available
    try:
        # Extract body pose parameters (joint angles) from SMPL data
        # smplx_data_1 contains 'body_pose' which is in axis-angle format
        pred_body_pose = torch.tensor(smplx_data_1['body_pose'], dtype=torch.float32)  # (T1, 63) for SMPL or (T1, 21*3)
        target_body_pose = torch.tensor(smplx_data_2['body_pose'], dtype=torch.float32)  # (T2, 63)
        
        # Align the body poses based on optimal time alignment
        pred_body_pose_optimal = pred_body_pose[best_start_idx:best_start_idx+T2]
        
        # Compute kinematic metrics for joint angles
        joint_angle_metrics = compute_kinematic_metrics(
            pred_body_pose_optimal, 
            target_body_pose[:T2], 
            fps=tgt_fps, 
            metric_name="joint_angle"
        )
        
        print(f"Joint Angle Velocity Error: {joint_angle_metrics['joint_angle_velocity_error_mean']:.6f} ± {joint_angle_metrics['joint_angle_velocity_error_std']:.6f} rad/s")
        print(f"Joint Angle Acceleration Error: {joint_angle_metrics['joint_angle_acceleration_error_mean']:.6f} ± {joint_angle_metrics['joint_angle_acceleration_error_std']:.6f} rad/s²")
        
        # Merge joint angle metrics into result
        result_optimal.update(joint_angle_metrics)
    except Exception as e:
        print(f"\nWarning: Could not compute joint angle metrics: {e}")
        # Add placeholder values
        result_optimal.update({
            'joint_angle_velocity_error_mean': None,
            'joint_angle_velocity_error_std': None,
            'joint_angle_acceleration_error_mean': None,
            'joint_angle_acceleration_error_std': None,
        })

    # ========== DTW COMPUTATION (OPTIONAL, ON OPTIMAL CUT) ==========
    if args.compute_dtw:
        print(f"\n{'='*60}")
        print("DTW-based Alignment (on Optimal Cut)")
        print(f"{'='*60}")
        
        # Use only the optimally aligned segment for DTW
        pred_j3d_cut = pred_j3d[best_start_idx:best_start_idx+T2]
        target_j3d_cut = target_j3d
        
        # 1) Build pairwise PA-MPJPE cost matrix (m) for the cut segment
        print("\nStep 1: Building pairwise PA-MPJPE cost matrix for DTW (on optimal cut)...")
        print(f"  Pred cut: frames {best_start_idx} to {best_start_idx+T2} (length: {T2})")
        print(f"  Target: frames 0 to {T2} (length: {T2})")
        
        cost_matrix_start = time.time()
        C = compute_pairwise_pa_mpjpe_matrix(pred_j3d_cut, target_j3d_cut, pelvis_idxs=[1, 2], device=pred_j3d.device, chunk=None, show_progress=True)
        cost_matrix_time = time.time() - cost_matrix_start
        print(f"Cost matrix shape: {C.shape}")
        print(f"Cost matrix computation time: {cost_matrix_time:.2f} seconds")
        print(f"Average time per pred frame: {cost_matrix_time/T2*1000:.2f} ms")

        # 2) Run DTW (optionally set a Sakoe-Chiba band, e.g., band=30)
        print("\nStep 2: Running DTW algorithm...")
        dtw_start = time.time()
        total_cost, path, D = dtw_from_cost_matrix(C, band=None)
        dtw_time = time.time() - dtw_start
        
        # Vectorized computation of average cost along path
        path_indices = np.array(path)
        avg_cost_along_path = C[path_indices[:, 0], path_indices[:, 1]].mean()

        print(f"DTW computation time: {dtw_time:.2f} seconds")
        print(f"Total DTW time (cost matrix + algorithm): {cost_matrix_time + dtw_time:.2f} seconds")

        print(f"\n{'='*60}")
        print("DTW Results:")
        print(f"{'='*60}")
        print(f"Total DTW cost: {total_cost:.6f} (sum of PA-MPJPE along path)")
        print(f"Path length: {len(path)}")
        print(f"Average PA-MPJPE along DTW path: {avg_cost_along_path:.6f} m")

        # Optional: show best brute-force shift vs DTW average
        print(f"\n{'='*60}")
        print("Comparison: Brute-force vs DTW")
        print(f"{'='*60}")
        print(f"Best brute-force PA-MPJPE (no warping): {best_pa_mpjpe:.6f} m")
        print(f"DTW average PA-MPJPE (with warping):    {avg_cost_along_path:.6f} m")
        improvement = ((best_pa_mpjpe - avg_cost_along_path) / best_pa_mpjpe * 100) if best_pa_mpjpe > 0 else 0
        print(f"Improvement: {improvement:.2f}%")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("DTW computation skipped (use --compute_dtw to enable)")
        print(f"{'='*60}")
        # Set default values for metrics that won't be computed
        avg_cost_along_path = None
        total_cost = None
        path = None

    # ========== SAVE RESULTS ==========
    # Save results to a text file in the same directory as gvhmr_1
    gvhmr_1_path = pathlib.Path(args.gvhmr_1)
    gvhmr_2_path = pathlib.Path(args.gvhmr_2)
    output_dir = gvhmr_1_path.parent
    output_filename = f"alignment_results.txt"
    output_path = output_dir / output_filename
    
    # Calculate timing information
    start_frame = best_start_idx
    end_frame = best_start_idx + T2
    start_time_sec = start_frame / tgt_fps
    end_time_sec = end_frame / tgt_fps
    duration_sec = (end_frame - start_frame) / tgt_fps
    
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_path}")
    print(f"{'='*60}")
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Motion Alignment Comparison Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Input Files:\n")
        f.write(f"  GVHMR 1: {args.gvhmr_1}\n")
        f.write(f"  GVHMR 2: {args.gvhmr_2}\n\n")
        
        f.write("Sequence Information:\n")
        f.write(f"  Frames in GVHMR 1 (T1): {T1}\n")
        f.write(f"  Frames in GVHMR 2 (T2): {T2}\n")
        f.write(f"  Target FPS: {tgt_fps}\n\n")
        
        f.write("="*60 + "\n")
        f.write("BRUTE-FORCE TIME ALIGNMENT (No Warping)\n")
        f.write("="*60 + "\n")
        f.write(f"Best start_idx: {best_start_idx}\n")
        f.write(f"Start time: {start_time_sec:.3f} seconds\n")
        f.write(f"End time: {end_time_sec:.3f} seconds\n")
        f.write(f"Duration: {duration_sec:.3f} seconds\n")
        f.write(f"PA-MPJPE: {best_pa_mpjpe:.6f} m\n")
        f.write(f"MPJPE: {best_mpjpe:.6f} m\n")
        f.write(f"Position Velocity Error: {result_optimal['position_velocity_error_mean']:.6f} ± {result_optimal['position_velocity_error_std']:.6f} m/s\n")
        f.write(f"Position Acceleration Error: {result_optimal['position_acceleration_error_mean']:.6f} ± {result_optimal['position_acceleration_error_std']:.6f} m/s²\n")
        
        # Add joint angle metrics if available
        if result_optimal.get('joint_angle_velocity_error_mean') is not None:
            f.write(f"Joint Angle Velocity Error: {result_optimal['joint_angle_velocity_error_mean']:.6f} ± {result_optimal['joint_angle_velocity_error_std']:.6f} rad/s\n")
            f.write(f"Joint Angle Acceleration Error: {result_optimal['joint_angle_acceleration_error_mean']:.6f} ± {result_optimal['joint_angle_acceleration_error_std']:.6f} rad/s²\n")
        
        f.write("="*60 + "\n")
        f.write("DTW Result\n")
        f.write("="*60 + "\n")
        if avg_cost_along_path is not None:
            f.write(f"PA-MPJPE (DTW): {avg_cost_along_path:.6f} m\n")
        else:
            f.write("DTW computation skipped\n")
    
    print(f"✓ Results saved successfully to: {output_path}")
    
    # ========== SAVE CSV RESULTS ==========
    csv_output_path = args.csv_output
    if csv_output_path is None:
        csv_output_path = output_dir / "alignment_results.csv"
    else:
        csv_output_path = pathlib.Path(csv_output_path)
    
    print(f"\n{'='*60}")
    print(f"Saving CSV results to: {csv_output_path}")
    print(f"{'='*60}")
    
    # Prepare metrics dictionary
    metrics = {
        'gvhmr_1': str(args.gvhmr_1),
        'gvhmr_2': str(args.gvhmr_2),
        'frames_gvhmr_1': T1,
        'frames_gvhmr_2': T2,
        'target_fps': tgt_fps,
        'best_start_idx': best_start_idx,
        'best_end_idx': end_frame,
        'start_time_sec': start_time_sec,
        'end_time_sec': end_time_sec,
        'duration_sec': duration_sec,
        'pa_mpjpe': best_pa_mpjpe,
        'mpjpe': best_mpjpe,
        'position_velocity_error_mean': result_optimal['position_velocity_error_mean'],
        'position_velocity_error_std': result_optimal['position_velocity_error_std'],
        'position_acceleration_error_mean': result_optimal['position_acceleration_error_mean'],
        'position_acceleration_error_std': result_optimal['position_acceleration_error_std'],
        'joint_angle_velocity_error_mean': result_optimal.get('joint_angle_velocity_error_mean', ''),
        'joint_angle_velocity_error_std': result_optimal.get('joint_angle_velocity_error_std', ''),
        'joint_angle_acceleration_error_mean': result_optimal.get('joint_angle_acceleration_error_mean', ''),
        'joint_angle_acceleration_error_std': result_optimal.get('joint_angle_acceleration_error_std', ''),
        'pa_mpjpe_dtw': avg_cost_along_path if avg_cost_along_path is not None else '',
        'dtw_total_cost': total_cost if total_cost is not None else '',
        'dtw_path_length': len(path) if path is not None else '',
    }
    # Overwrite CSV file (use 'w' mode instead of 'a')
    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    print(f"✓ CSV results overwritten successfully to: {csv_output_path}")
    
    # ========== CUT VIDEOS (OPTIONAL) ==========
    if args.cut_videos:
        print(f"\n{'='*60}")
        print("Cutting Videos Based on Optimal Alignment")
        print(f"{'='*60}")
        
        print(f"\nVideo cutting parameters:")
        print(f"  Start frame: {start_frame}")
        print(f"  End frame: {end_frame}")
        print(f"  Start time: {start_time_sec:.3f} seconds")
        print(f"  Duration: {duration_sec:.3f} seconds")
        print(f"  FPS: {tgt_fps}")
        
        # Get the directory where gvhmr_1 file is located
        gvhmr_1_dir = pathlib.Path(args.gvhmr_1).parent
        
        # Process each video
        for video_name in args.video_names:
            input_video = gvhmr_1_dir / video_name
            
            if not input_video.exists():
                print(f"\n⚠ Warning: Video file not found: {input_video}")
                continue
            
            # Create output filename with _cut suffix
            output_video = gvhmr_1_dir / f"{input_video.stem}_cut{input_video.suffix}"
            
            print(f"\nProcessing {video_name}...")
            print(f"  Input: {input_video}")
            print(f"  Output: {output_video}")
            
            success = cut_video_with_ffmpeg(input_video, output_video, start_frame, end_frame, tgt_fps)
            
            if success:
                print(f"  ✓ Successfully created: {output_video}")
            else:
                print(f"  ✗ Failed to create: {output_video}")
        
        print(f"\n{'='*60}")
        print("Video Cutting Complete")
        print(f"{'='*60}")
    else:
        print(f"\nVideo cutting skipped (use --cut_videos to enable)")
