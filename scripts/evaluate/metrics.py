import numpy as np

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


def compute_dtw(recorded_pos, reference_pos, normalize=True):
    """Compute DTW distance based on MPJPE.
    
    Args:
        recorded_pos: (T1, num_joints, 3) numpy array
        reference_pos: (T2, num_joints, 3) numpy array
        normalize (bool): If True, returns the 'per-step' average error (mm).
                          If False, returns the cumulative cost (mm * frames).

    Returns:
        dtw_dist: The DTW cost (normalized or cumulative).
    """
    T1, num_joints, _ = recorded_pos.shape
    T2, _, _ = reference_pos.shape

    # --- 1. Compute Cost Matrix (Vectorized) ---
    # Cost[i, j] = MPJPE between recorded[i] and reference[j]
    cost_matrix = np.zeros((T1, T2))

    for i in range(T1):
        # Broadcast difference: (T2, J, 3)
        diff = reference_pos - recorded_pos[i]
        # Euclidean dist per joint: (T2, J)
        joint_errors = np.linalg.norm(diff, axis=2)
        # MPJPE for this frame pair: (T2,)
        cost_matrix[i, :] = np.mean(joint_errors, axis=1)

    # --- 2. DTW Accumulation (Dynamic Programming) ---
    acc_cost = np.full((T1 + 1, T2 + 1), np.inf)
    acc_cost[0, 0] = 0

    # Fill accumulation matrix
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = cost_matrix[i-1, j-1]
            acc_cost[i, j] = cost + min(
                acc_cost[i-1, j-1],  # Match
                acc_cost[i-1, j],    # Insertion
                acc_cost[i, j-1]     # Deletion
            )

    total_cost = acc_cost[T1, T2]

    if not normalize:
        return total_cost

    # --- 3. Normalization (Backtracking) ---
    # To normalize correctly (Average MPJPE), we divide by the length of the
    # optimal path, not just T1 or T2.
    path_len = 0
    i, j = T1, T2
    
    # Backtrack from bottom-right to top-left to count steps
    while i > 0 and j > 0:
        path_len += 1
        # Look at the three neighbors to find where we came from
        diag = acc_cost[i-1, j-1]
        up   = acc_cost[i-1, j]
        left = acc_cost[i, j-1]
        
        # Greedily move to the neighbor with the lowest accumulated cost
        # (This logic must match the min() logic used in the forward pass)
        best = min(diag, up, left)
        
        if best == diag:
            i -= 1; j -= 1
        elif best == up:
            i -= 1
        else:
            j -= 1

    # Average error in mm
    return total_cost / path_len


def compute_tlcc(recorded_pos, reference_pos, fps=30.0):
    """
    Compute Time-Lagged Cross-Correlation (TLCC) and Phase Lag.
    
    This metric separates 'temporal delay' from 'morphological error'.
    It answers: "If we shift the signals to align in time, how well do they match?"
    
    Args:
        recorded_pos: (T1, num_joints, 3) - The robot/recorded motion
        reference_pos: (T2, num_joints, 3) - The target/reference motion
        fps: Frames per second (float) - Used to calculate lag in milliseconds
        
    Returns:
        peak_corr (float): The maximum Pearson correlation coefficient [-1, 1].
                           1.0 = Perfect shape match.
        phase_lag_ms (float): The temporal lag in milliseconds. 
                              Positive (+) = Robot is DELAYED (Late).
                              Negative (-) = Robot is AHEAD (Early).
    """
    # 1. Flatten the spatial dimensions (T, J, 3) -> (T, J*3)
    # We treat the whole pose as a single high-dimensional vector at each timestep.
    T1, num_joints, _ = recorded_pos.shape
    T2 = reference_pos.shape[0]
    
    flat_rec = recorded_pos.reshape(T1, -1)
    flat_ref = reference_pos.reshape(T2, -1)
    
    # 2. Normalize features (Z-score) to compute Pearson Correlation
    # Subtract mean and divide by std deviation for each dimension (joint coordinate)
    # This ensures the magnitude of movement doesn't bias the correlation, only the 'shape'.
    rec_centered = flat_rec - np.mean(flat_rec, axis=0)
    ref_centered = flat_ref - np.mean(flat_ref, axis=0)
    
    rec_std = np.std(flat_rec, axis=0)
    ref_std = np.std(flat_ref, axis=0)
    
    # Avoid division by zero for static joints
    rec_std[rec_std == 0] = 1.0
    ref_std[ref_std == 0] = 1.0
    
    rec_norm = rec_centered / rec_std
    ref_norm = ref_centered / ref_std

    # 3. Compute Cross-Correlation averaged across all dimensions
    # We compute the correlation for each coordinate and take the mean profile.
    # This is robust against one specific joint having high variance.
    num_features = flat_rec.shape[1]
    total_corr = np.zeros(T1 + T2 - 1)
    
    for i in range(num_features):
        # mode='full' returns the convolution at all possible overlaps
        total_corr += np.correlate(rec_norm[:, i], ref_norm[:, i], mode='full')
    
    # Average across joints/coordinates and normalize by sequence length
    # Note: Strictly speaking, Pearson divides by N. In 'full' mode, N varies, 
    # but for finding the peak in similar-length sequences, dividing by the 
    # max length is a standard approximation for the coefficient.
    avg_corr = total_corr / (num_features * max(T1, T2))
    
    # 4. Find the Peak and the Lag
    peak_idx = np.argmax(avg_corr)
    peak_corr = avg_corr[peak_idx]
    
    # 5. Convert Index to Time Lag
    # In np.correlate(Rec, Ref), the index 0 corresponds to Rec sliding 
    # all the way to the left of Ref. The "zero lag" center is at index len(Ref) - 1.
    # Formula: Lag = Peak_Index - (Length_Reference - 1)
    shift_frames = peak_idx - (T2 - 1)
    phase_lag_ms = (shift_frames / fps) * 1000.0
    
    return peak_corr, phase_lag_ms