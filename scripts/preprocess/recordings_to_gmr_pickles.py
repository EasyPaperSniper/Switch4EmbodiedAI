from tqdm import tqdm
import pickle
import numpy as np

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

# recording_paths = [
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Unstoppable/Unstoppable_Online_Reference.txt",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Pink_Venom/Pink_Venom_Online_Reference.txt",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Padam_Padam/Padam_Padam_Online_Reference.txt",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Old_Town_Road/Old_Town_Road_Online_Reference.txt",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Heart_Of_Glass/Heart_Of_Glass_Online_Reference.txt",
# ]
# output_gmr_paths = [
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Unstoppable/Unstoppable_Online_Reference_gmr.pkl",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Pink_Venom/Pink_Venom_Online_Reference_gmr.pkl",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Padam_Padam/Padam_Padam_Online_Reference_gmr.pkl",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Old_Town_Road/Old_Town_Road_Online_Reference_gmr.pkl",
#     "/home/jkim3662/Videos/Switch4EAI/ReferenceSwitchRecordings_GMR/online/Heart_Of_Glass/Heart_Of_Glass_Online_Reference_gmr.pkl",
# ]
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
    dof_pos,
    vel_threshold=0.03,
    gap_tolerance=250,
    padding=30,
    plot_idle=False
):
    vel = np.max(np.abs(np.diff(dof_pos, axis=0)), axis=1)
    vel = np.concatenate([[0], vel])

    active = vel > vel_threshold
    T = len(active)

    # --- Find raw segments ---segments = []
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
        return dof_pos

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
    start = max(0, start - padding)
    end = min(T, end + padding)

    trimmed = dof_pos[start:end]

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

def resample_fps(arr, old_fps=50, new_fps=30):
    """
    Resample a time-series array from old_fps -> new_fps using linear interpolation.
    arr: (T, D)
    returns: (T_new, D)
    """
    T = arr.shape[0]
    duration = T / old_fps  # seconds

    t_old = np.linspace(0, duration, T, endpoint=False)
    T_new = int(duration * new_fps)
    t_new = np.linspace(0, duration, T_new, endpoint=False)

    # Interpolate each dimension independently
    arr_new = np.zeros((T_new, arr.shape[1]))
    for d in range(arr.shape[1]):
        arr_new[:, d] = np.interp(t_new, t_old, arr[:, d])

    return arr_new

def resample_fps_from_timestamps(arr, timestamps, new_fps=30):
    # from timestamps, resample to new fps
    T = arr.shape[0]
    duration = timestamps[-1] - timestamps[0] # seconds
    t_old = timestamps
    T_new = int(duration * new_fps)
    t_new = np.linspace(timestamps[0], timestamps[-1], T_new, endpoint=False)
    arr_new = np.zeros((T_new, arr.shape[1]))
    for d in range(arr.shape[1]):
        arr_new[:, d] = np.interp(t_new, t_old, arr[:, d])
    return arr_new

for i, recording in tqdm(enumerate(recordings)):
    dof_pos = recording[:, 1:24]  # Extract DOF positions from columns 1 to 24
    
    gmr_dof_pos = dof_pos_rec_to_gmr(dof_pos)
    NEW_FPS = 30
    gmr_dof_pos = resample_fps_from_timestamps(gmr_dof_pos, recording[:, 0], new_fps=NEW_FPS)
    # OLD_FPS = 50
    # gmr_dof_pos = resample_fps(gmr_dof_pos, old_fps=OLD_FPS, new_fps=NEW_FPS)

    gmr_dof_pos = trim_idle_dofs(gmr_dof_pos, padding=NEW_FPS)
    n_frames = gmr_dof_pos.shape[0]

    gmr_root_pos = np.zeros((n_frames, 3))  # Set root position to zeros
    gmr_root_pos[:, 2] = 1.0  # Set a constant height for the root (e.g., z=1.0)
    gmr_rot_wxyz = np.zeros((n_frames, 4))
    gmr_rot_wxyz[:, 0] = 1.0  # Set w component to 1 (no rotation)
    gmr_rot_xyzw = gmr_rot_wxyz[:, [1, 2, 3, 0]]  # Convert wxyz to xyzw if needed

    pose_data = {
        'fps': 30,
        'root_pos': gmr_root_pos,  # Columns 1 to 4 for root position
        'root_rot': gmr_rot_xyzw,  # Columns 4 to 7 for root rotation
        'dof_pos': gmr_dof_pos,
        'local_body_pos': None,  # Not available in the recording
        'link_body_list': None,  # Not available in the recording
    }
    
    print(f"Saving GMR pickle to: {output_gmr_paths[i]}"
          f" with {n_frames} frames. corresponding to {n_frames/NEW_FPS:.2f} seconds. ")
    outfile_path = output_gmr_paths[i]
    with open(outfile_path, "wb") as f:
        pickle.dump(pose_data, f)

