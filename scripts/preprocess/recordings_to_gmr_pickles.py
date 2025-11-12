from tqdm import tqdm
import pickle
import numpy as np

# Add parent directory to path
import sys
import pathlib
HERE = pathlib.Path(__file__).parent
sys.path.append(str(HERE / ".." / ".."))


# from scripts.evaluate.data.gmt_sim_paths import txt_paths as recording_paths, gmr_paths as output_gmr_paths
from scripts.evaluate.data.twist_sim_paths import txt_paths as recording_paths, gmr_paths as output_gmr_paths
# from scripts.evaluate.data.any2track_sim_paths import txt_paths as recording_paths, gmr_paths as output_gmr_paths
# from scripts.evaluate.data.gmt_paths import txt_paths as recording_paths, gmr_paths as output_gmr_paths
# from scripts.evaluate.data.twist_paths import txt_paths as recording_paths, gmr_paths as output_gmr_paths
# from scripts.evaluate.data.any2track_paths import txt_paths as recording_paths, gmr_paths as output_gmr_paths

APPLY_PADDING = True
SONG_INFO = {
    "Old_Town_Road": 161,
    "Heart_Of_Glass": 216,
    "Unstoppable": 204,
    "Padam_Padam": 149,
    "Pink_Venom": 178,
}

recordings = [
    np.loadtxt(recording_path, delimiter=',') for recording_path in recording_paths
]

# Helper function to get song name from path
from pathlib import Path
def get_song_name(path):
    """Extract song name from GMR file path."""
    filename = Path(path).stem.replace("_poses", "")
    for song in SONG_INFO.keys():
        if filename.startswith(song):
            return song
    return None

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
    gap_tolerance=100,
    padding=30,
    plot_idle=False
):
    vel = np.max(np.abs(np.diff(dof_pos, axis=0)), axis=1)
    vel = np.concatenate([[0], vel])

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

# for i, recording in tqdm(enumerate(recordings)):
#     dof_pos = recording[:, 1:24]  # Extract DOF positions from columns 1 to 24
    
#     gmr_dof_pos = dof_pos_rec_to_gmr(dof_pos)
#     NEW_FPS = 30
#     gmr_dof_pos = resample_fps_from_timestamps(gmr_dof_pos, recording[:, 0], new_fps=NEW_FPS)
#     # OLD_FPS = 50
#     # gmr_dof_pos = resample_fps(gmr_dof_pos, old_fps=OLD_FPS, new_fps=NEW_FPS)

#     gmr_dof_pos = trim_idle_dofs(gmr_dof_pos, padding=NEW_FPS)
#     n_original_frames = gmr_dof_pos.shape[0]

#     # --- Save unpadded version first ---
#     gmr_root_pos = np.zeros((n_original_frames, 3))
#     gmr_root_pos[:, 2] = 1.0
#     gmr_rot_wxyz = np.zeros((n_original_frames, 4))
#     gmr_rot_wxyz[:, 0] = 1.0
#     gmr_rot_xyzw = gmr_rot_wxyz[:, [1, 2, 3, 0]]

#     pose_data_unpadded = {
#         'fps': NEW_FPS,
#         'root_pos': gmr_root_pos,
#         'root_rot': gmr_rot_xyzw,
#         'dof_pos': gmr_dof_pos,
#         'local_body_pos': None,
#         'link_body_list': None,
#     }

#     outfile_unpadded = output_gmr_paths[i]
#     print(f"Saving unpadded GMR pickle to: {outfile_unpadded} ({n_original_frames} frames)")
#     with open(outfile_unpadded, "wb") as f:
#         pickle.dump(pose_data_unpadded, f)

#     # --- Apply padding if requested ---
#     if APPLY_PADDING:
#         n_pad_frames = SONG_INFO[get_song_name(output_gmr_paths[i])] * NEW_FPS
#         gmr_dof_pos_padded = gmr_dof_pos
#         if gmr_dof_pos_padded.shape[0] < n_pad_frames:
#             n_missing = n_pad_frames - gmr_dof_pos_padded.shape[0]
#             gmr_dof_pos_padded = np.pad(
#                 gmr_dof_pos_padded,
#                 ((0, n_missing), (0, 0)),
#                 mode='edge'
#             )
#         else:
#             gmr_dof_pos_padded = gmr_dof_pos_padded[:n_pad_frames]

#         n_frames_padded = gmr_dof_pos_padded.shape[0]
#         gmr_root_pos_padded = np.zeros((n_frames_padded, 3))
#         gmr_root_pos_padded[:, 2] = 1.0
#         gmr_rot_wxyz_padded = np.zeros((n_frames_padded, 4))
#         gmr_rot_wxyz_padded[:, 0] = 1.0
#         gmr_rot_xyzw_padded = gmr_rot_wxyz_padded[:, [1, 2, 3, 0]]

#         pose_data_padded = {
#             'fps': NEW_FPS,
#             'root_pos': gmr_root_pos_padded,
#             'root_rot': gmr_rot_xyzw_padded,
#             'dof_pos': gmr_dof_pos_padded,
#             'local_body_pos': None,
#             'link_body_list': None,
#         }

#         outfile_padded = output_gmr_paths[i].replace(".pkl", "-padded.pkl")
#         print(f"Saving padded GMR pickle to: {outfile_padded} ({n_frames_padded} frames)")
#         with open(outfile_padded, "wb") as f:
#             pickle.dump(pose_data_padded, f)

#         # Save padding info
#         n_original = n_original_frames
#         n_padded = max(0, n_pad_frames - n_original)
#         info_text = (
#             f"Original frames: {n_original}\n"
#             f"Padded frames: {n_padded}\n"
#             f"Final  frames: {n_frames_padded}\n"
#             f"Target total frames: {n_pad_frames}\n"
#             f"Target duration (s): {SONG_INFO[get_song_name(output_gmr_paths[i])]}\n"
#             f"FPS: {NEW_FPS}\n"
#         )
#         info_path = outfile_padded.replace(".pkl", "_padinfo.txt")
#         with open(info_path, "w") as f:
#             f.write(info_text)

for i, recording in tqdm(enumerate(recordings)):
    # --- Step 1. Load raw recording ---
    n_raw_frames = recording.shape[0]
    dof_pos = recording[:, 1:24]  # Extract DOF positions
    print(f"\n[{i}] Raw frames: {n_raw_frames}")

    # --- Step 2. Convert and resample ---
    gmr_dof_pos = dof_pos_rec_to_gmr(dof_pos)
    NEW_FPS = 30
    gmr_dof_pos = resample_fps_from_timestamps(gmr_dof_pos, recording[:, 0], new_fps=NEW_FPS)

    # --- Step 3. Trim idle motion ---
    gmr_dof_pos = trim_idle_dofs(gmr_dof_pos, padding=NEW_FPS)
    n_trimmed_frames = gmr_dof_pos.shape[0]
    print(f"Trimmed frames: {n_trimmed_frames}")

    # --- Step 4. Save unpadded GMR ---
    gmr_root_pos = np.zeros((n_trimmed_frames, 3))
    gmr_root_pos[:, 2] = 1.0
    gmr_rot_wxyz = np.zeros((n_trimmed_frames, 4))
    gmr_rot_wxyz[:, 0] = 1.0
    gmr_rot_xyzw = gmr_rot_wxyz[:, [1, 2, 3, 0]]

    pose_data_unpadded = {
        'fps': NEW_FPS,
        'root_pos': gmr_root_pos,
        'root_rot': gmr_rot_xyzw,
        'dof_pos': gmr_dof_pos,
        'local_body_pos': None,
        'link_body_list': None,
    }

    outfile_unpadded = output_gmr_paths[i]
    print(f"Saving unpadded GMR pickle to: {outfile_unpadded} ({n_trimmed_frames} frames)")
    with open(outfile_unpadded, "wb") as f:
        pickle.dump(pose_data_unpadded, f)

    # --- Step 5. Apply padding (if requested) ---
    if APPLY_PADDING:
        song_name = get_song_name(output_gmr_paths[i])
        n_pad_frames = SONG_INFO[song_name] * NEW_FPS

        if n_trimmed_frames < n_pad_frames:
            n_missing = n_pad_frames - n_trimmed_frames
            gmr_dof_pos_padded = np.pad(
                gmr_dof_pos, ((0, n_missing), (0, 0)), mode='edge'
            )
        else:
            n_missing = 0
            gmr_dof_pos_padded = gmr_dof_pos

        n_padded_frames = gmr_dof_pos_padded.shape[0]

        gmr_root_pos_padded = np.zeros((n_padded_frames, 3))
        gmr_root_pos_padded[:, 2] = 1.0
        gmr_rot_wxyz_padded = np.zeros((n_padded_frames, 4))
        gmr_rot_wxyz_padded[:, 0] = 1.0
        gmr_rot_xyzw_padded = gmr_rot_wxyz_padded[:, [1, 2, 3, 0]]

        pose_data_padded = {
            'fps': NEW_FPS,
            'root_pos': gmr_root_pos_padded,
            'root_rot': gmr_rot_xyzw_padded,
            'dof_pos': gmr_dof_pos_padded,
            'local_body_pos': None,
            'link_body_list': None,
        }

        outfile_padded = output_gmr_paths[i].replace(".pkl", "-padded.pkl")
        print(f"Saving padded GMR pickle to: {outfile_padded} ({n_padded_frames} frames)")
        with open(outfile_padded, "wb") as f:
            pickle.dump(pose_data_padded, f)

        # --- Step 6. Save info summary --- 
        n_added_padding = max(0, n_pad_frames - n_trimmed_frames)
        info_text = (
            f"Original raw frames: {n_raw_frames}\n"
            f"Trimmed active frames: {n_trimmed_frames}\n"
            f"Padded frames added: {n_added_padding}\n"
            f"Padded total frames: {n_padded_frames}\n"
            f"Target duration (s): {SONG_INFO[song_name]}\n"
            f"FPS: {NEW_FPS}\n"
        )
        info_path = outfile_padded.replace(".pkl", "_padinfo.txt")
        with open(info_path, "w") as f:
            f.write(info_text)

        print(f"✅ Summary for {song_name}:")
        print(info_text)
