from tqdm import tqdm
import pickle
import numpy as np

# Dictionary mapping filename -> frames to pad at the beginning (repeating the first frame)
start_pads = {
    # "Baby_Shark": 434,
    # "Heart_Of_Glass": 238,
    # "Old_Town_Road": 123,
    # "Padam_Padam": 735,
    # "Soy_Yo": 470,
    # "Unstoppable": 365,
    "Pink_Venom": 450,
}

recording_paths = [
    # Unstoppable
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_offline-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_offline-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_offline-3.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_online-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_online-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_online-3.txt",
    # Pink_Venom
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_offline-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_offline-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_offline-3.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_online-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_online-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_online-3.txt",
    # Padam_Padam
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_offline-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_offline-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_offline-3.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_online-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_online-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_online-3.txt",
    # Old_Town_Road
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_offline-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_offline-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_offline-3.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_online-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_online-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_online-3.txt",
    # Heart_Of_Glass
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_offline-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_offline-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_offline-3.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_online-1.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_online-2.txt",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_online-3.txt",
]
output_gmr_paths = [
    # Unstoppable
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_offline-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_offline-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_offline-3_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_online-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_online-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Unstoppable/Unstoppable_online-3_gmr.pkl",
    # Pink_Venom
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_offline-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_offline-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_offline-3_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_online-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_online-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Pink_Venom/Pink_Venom_online-3_gmr.pkl",
    # Padam_Padam
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_offline-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_offline-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_offline-3_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_online-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_online-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Padam_Padam/Padam_Padam_online-3_gmr.pkl",
    # Old_Town_Road
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_offline-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_offline-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_offline-3_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_online-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_online-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Old_Town_Road/Old_Town_Road_online-3_gmr.pkl",
    # Heart_Of_Glass
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_offline-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_offline-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_offline-3_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_online-1_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_online-2_gmr.pkl",
    "/home/jkim3662/Videos/Switch4EAI/Switch4EAI_Collaborators_Archive/TWIST/RobotTrajectoryRecord/Heart_Of_Glass/Heart_Of_Glass_online-3_gmr.pkl",
]
recordings = [
    np.loadtxt(recording_path, delimiter=',') for recording_path in recording_paths
]

def dof_pos_rec_to_gmr(dof_pos_rec):
    # dof_pos_txt is a numpy array of shape (num_frames, 23)
    # Need to change it to GMR format of shape (num_frames, 29)
    # dof_idx_twist_from_gmr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
    # 15, 16, 17, 18, 
    # 22, 23, 24, 25]

    gmr_dof_pos = []
    for frame in dof_pos_rec:
        gmr_frame = np.zeros(29)  # GMR has 29 DOFs
        # Map existing DOFs
        gmr_frame[:15] = frame[:15]      # lowerbody(6+6+3)
        gmr_frame[15:19] = frame[15:19]  # left_arm(4)
        # gmr_frame[19:22] = 0             # left_wrist(3) - set to 0
        gmr_frame[22:26] = frame[19:23]  # right_arm(4)
        # gmr_frame[26:29] = 0             # right_wrist(3) - set to 0
        gmr_dof_pos.append(gmr_frame)

    return np.array(gmr_dof_pos)

import numpy as np

def trim_idle_dofs(
    dof_recording,
    vel_threshold=0.03,
    min_active_len=300,
    gap_tolerance=250,
    plot_idle=True
):
    vel = np.max(np.abs(np.diff(dof_recording, axis=0)), axis=1)
    vel = np.concatenate([[0], vel])
    vel[:200] = 0.0  # ignore setup jitter

    active = vel > vel_threshold
    T = len(active)

    # --- Find raw segments ---
    segments = []
    cur_start = None
    for i in range(T):
        if active[i] and cur_start is None:
            cur_start = i
        elif not active[i] and cur_start is not None:
            segments.append((cur_start, i))
            cur_start = None
    if cur_start is not None:
        segments.append((cur_start, T))
    
    if len(segments) == 0:
        print("No active segments found; returning full recording.")
        return dof_recording

    # --- Merge segments separated by small gaps ---
    merged = []
    cur_s, cur_e = segments[0]

    for s, e in segments[1:]:
        if s - cur_e <= gap_tolerance:  # gap small → merge
            cur_e = e
        else:  # gap large → finalize current segment
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Pick the longest final merged segment
    lengths = [(end - start) for start, end in merged]
    best_idx = np.argmax(lengths)
    start, end = merged[best_idx]

    # Expand slightly for smoother playback
    start = max(0, start - 50)
    end = min(T, end + 50)

    trimmed = dof_recording[start:end]

    if plot_idle:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14,4))
        plt.plot(vel, label='Velocity')
        plt.axhline(vel_threshold, color='r', linestyle='--', label='Threshold')
        plt.axvline(start, color='g', linestyle='--', label='Start Trim')
        plt.axvline(end, color='m', linestyle='--', label='End Trim')
        plt.title("Merged Active Dance Segment (Gap Tolerant)")
        plt.legend()
        plt.show()

    print(f"Trimmed: start={start}, end={end}, frames={end-start}")
    return trimmed


for i, recording in tqdm(enumerate(recordings)):
    dof_recording = recording[:, 1:24]  # Extract DOF positions from columns 1 to 24
    dof_recording = trim_idle_dofs(dof_recording)
    n_frames = dof_recording.shape[0]
    
    gmr_dof_pos = dof_pos_rec_to_gmr(dof_recording)
    gmr_root_pos = np.zeros((n_frames, 3))  # Set root position to zeros
    gmr_root_pos[:, 2] = 0.9  # Set a constant height for the root (e.g., z=0.9)
    gmr_rot_wxyz = recording[:, 24:28]  # Columns 4 to 7 for root rotation (wxyz)
    gmr_rot_wxyz = np.zeros_like(gmr_rot_wxyz)
    gmr_rot_wxyz[:, 0] = 1.0  # Set w component to 1 (no rotation)
    gmr_rot_xyzw = gmr_rot_wxyz[:, [1, 2, 3, 0]]  # Convert wxyz to xyzw if needed
    n_frames = gmr_dof_pos.shape[0]
    pose_data = {
        'fps': 50,
        'root_pos': gmr_root_pos,  # Columns 1 to 4 for root position
        'root_rot': gmr_rot_xyzw,  # Columns 4 to 7 for root rotation
        'dof_pos': gmr_dof_pos,
        'local_body_pos': None,  # Not available in the recording
        'link_body_list': None,  # Not available in the recording
    }
    
    print(f"Saving GMR pickle to: {output_gmr_paths[i]}"
          f" with {n_frames} frames. corresponding to {n_frames/50:.2f} seconds. ")
    outfile_path = output_gmr_paths[i]
    with open(outfile_path, "wb") as f:
        pickle.dump(pose_data, f)

