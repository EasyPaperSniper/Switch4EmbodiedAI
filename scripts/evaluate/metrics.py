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

