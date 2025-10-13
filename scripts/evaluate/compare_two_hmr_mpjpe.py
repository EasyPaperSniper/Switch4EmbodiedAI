import pathlib
import torch
import numpy as np
import argparse
import os
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
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).numpy()  # (frames,)


def compute_perjoint_jpe(S1, S2):
    # S1, S2: (frames, num_joints, 3)
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).numpy()  # (frames, num_joints)


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
    """
    assert pred_j3d.shape[0] == target_j3d.shape[0], "The number of frames in pred_j3d and target_j3d must be the same."

    # Send the values to torch tensors
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
    m2mm = 1000
    # Per-frame Procrustes alignment of pred_j3d to target_j3d
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d).mean() # (num_frames,) -> float
    mpjpe = compute_jpe(pred_j3d, target_j3d).mean() # (num_frames,) -> float
    
    perjoint_metrics = { # per-joint metrics in numpy array
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
    }
    return perjoint_metrics

# --- Helper Functions ---
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
    pred_j3d = pred_j3d[:T2]  # truncate to the shorter length
    
    result = compute_perjoint_metrics(pred_j3d, target_j3d, pelvis_idxs=[1,2])
    print("Sequence-level PA-MPJPE:", result['pa_mpjpe'].mean())
    print("Sequence-level MPJPE:", result['mpjpe'].mean())

