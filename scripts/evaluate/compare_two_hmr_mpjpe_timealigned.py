import pathlib
import torch
import numpy as np
import argparse
import os
import time
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
def compute_perjoint_metrics(pred_j3d, target_j3d, pelvis_idxs=[1, 2]):
    """
    seq1, seq2: (T, J, 3), (T, J, 3) in same joint order and units (e.g., m)
    Returns:
      {
        'pa_mpjpe': float, # mean per-joint position error after global PA
        'mpjpe': float,    # mean per-joint position error only with pelvis alignment
      }
    
    Note: For optimal performance, pass tensors directly to avoid repeated conversions.
    """
    assert pred_j3d.shape[0] == target_j3d.shape[0], "The number of frames in pred_j3d and target_j3d must be the same."

    # Send the values to torch tensors (only if not already tensors)
    if not torch.is_tensor(pred_j3d):
        pred_j3d = torch.tensor(pred_j3d, dtype=torch.float32)
    if not torch.is_tensor(target_j3d):
        target_j3d = torch.tensor(target_j3d, dtype=torch.float32)

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
    
    perjoint_metrics = { # per-joint metrics in numpy array
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
    }
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
    args = parser.parse_args()
    
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models" / "smplx"
    SMPLX_FOLDER = "assets/body_models"
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
    result_optimal = compute_perjoint_metrics(pred_j3d_optimal, target_j3d, pelvis_idxs=[1,2])
    print("\nFinal verification:")
    print(f"Sequence-level PA-MPJPE: {result_optimal['pa_mpjpe']:.6f}")
    print(f"Sequence-level MPJPE: {result_optimal['mpjpe']:.6f}")

    # ========== DTW COMPUTATION ==========
    print(f"\n{'='*60}")
    print("DTW-based Alignment")
    print(f"{'='*60}")
    
    # 1) Build pairwise PA-MPJPE cost matrix (m)
    print("\nStep 1: Building pairwise PA-MPJPE cost matrix for DTW...")
    cost_matrix_start = time.time()
    C = compute_pairwise_pa_mpjpe_matrix(pred_j3d, target_j3d, pelvis_idxs=[1, 2], device=pred_j3d.device, chunk=None, show_progress=True)
    cost_matrix_time = time.time() - cost_matrix_start
    print(f"Cost matrix shape: {C.shape}")
    print(f"Cost matrix computation time: {cost_matrix_time:.2f} seconds")
    print(f"Average time per pred frame: {cost_matrix_time/T1*1000:.2f} ms")

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
    print(f"Improvement: {((best_pa_mpjpe - avg_cost_along_path) / best_pa_mpjpe * 100):.2f}%")
    print(f"{'='*60}")
