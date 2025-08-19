import argparse
import pathlib
import os
import time
import datetime
import signal
import threading

import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
from scipy.spatial.transform import Rotation as R

from third_party.GMR.general_motion_retargeting import GeneralMotionRetargeting as GMR
from third_party.GMR.general_motion_retargeting import RobotMotionViewer
from third_party.GMR.general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

from Switch4EmbodiedAI.utils.helpers import get_args, parse_StreamModule_cfg, parse_MocapModule_cfg, parse_RetgtModule_cfg
from Switch4EmbodiedAI.modules.Mocap_Module import *
from Switch4EmbodiedAI.utils.helpers import signal_handler
from Switch4EmbodiedAI.utils.qpos_smoother import QposStreamSmoother



class GMR_RetgtModule():
    # Only support one character retargeting now!!!

    def __init__(self, config):
        
        self.config = config
        self.retgt_module = None
        self.robot_motion_viewer = None
        self.smoother = QposStreamSmoother(window_size=10)
        # Decoupled stepping state
        self._latest_qpos = None
        self._qpos_lock = threading.Lock()
        self._step_thread = None
        self._step_stop = threading.Event()

        self.body_model = smplx.create(
            self.config.smplx_file,
            "smplx",
            gender="NEUTRAL",
            use_pca=False,
        )

        self.save_retgt_buffer = []


    def reset(self):
        self.retgt_module = None
        self.robot_motion_viewer = None
        self._latest_qpos = None
        self._step_thread = None
        self._step_stop = threading.Event()



    def init_module(self, actual_human_height):
        self.smoother.reset()
        self.retgt_module = GMR(
            src_human="smplx",
            tgt_robot=self.config.robot,
            actual_human_height=actual_human_height,
            )
        # Start decoupled stepping thread (viewer will be created inside that thread)
        if self._step_thread is None and self.config.viz_retgt:
            self._step_stop.clear()
            self._step_thread = threading.Thread(target=self._stepper_loop, daemon=True)
            self._step_thread.start()
        

        

    def retarget(self, input_data, smooth_output=True):
        # retarget the input data to the robot motion
        smplx_output, actual_human_height = self.load_smplx_file(input_data)
        smplx_data_frames = self.parse_smplx_output(smplx_output ,self.body_model)
        if self.retgt_module is None:
            self.init_module(actual_human_height)
        qpos = self.retgt_module.retarget(smplx_data_frames)
        if smooth_output:
            qpos = self.smoother.add(qpos)
        # Publish latest qpos for decoupled stepping
        with self._qpos_lock:
            self._latest_qpos = qpos.copy()
        
        return qpos
    
    def close(self):
        """Close the robot motion viewer if it exists."""
        # Stop stepper thread first
        if self._step_thread is not None:
            self._step_stop.set()
            try:
                self._step_thread.join(timeout=2.0)
            except Exception:
                pass
            self._step_thread = None

        if self.robot_motion_viewer is not None:
            self.robot_motion_viewer.close()
            print("Robot motion viewer closed.")
        # Save retargeted trajectory if requested
        try:
            if getattr(self.config, 'save_retgt', False) and len(self.save_retgt_buffer) > 0:
                buffer_arr = np.asarray(self.save_retgt_buffer)
                root_pos = buffer_arr[:, :3]
                root_rot = buffer_arr[:, 3:7]  # already converted to xyzw when stored
                dof_pos = buffer_arr[:, 7:]
                motion_data = {
                    'root_pos': root_pos,
                    'root_rot': root_rot,
                    'dof_pos': dof_pos,
                }
                ts = datetime.datetime.now().strftime('%m-%d_%H-%M')
                save_dir = self.config.save_path if self.config.save_path is not None else '.'
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception:
                    pass
                file_path = os.path.join(save_dir, f"{self.config.robot}_{ts}.npz")
                np.savez(file_path, **motion_data)
                print(f"Saved retargeted trajectory to {file_path}, {buffer_arr.shape[0]} frames")
        except Exception as e:
            print(f"Warning: failed to save retargeted trajectory: {e}")
        self.reset()



    def load_smplx_file(self, smplx_data):
        num_frames = smplx_data["body_pose"].shape[0]
        # if len(smplx_data["smpl_betas"].shape)==1:
        #     human_height = 1.66 + 0.1 * smplx_data["smpl_betas"][0]
        # else:
        #     human_height = 1.66 + 0.1 * smplx_data["smpl_betas"][0, 0]
        
        # # adjust the camera height
        # if self.retgt_module is None:
        #     self.mocap_delta_pos = smplx_data["transl"][0].copy()
        #     self.mocap_delta_pos[2] = smplx_data["transl"][0,2] - human_height * 0.5 -0.05
        # smplx_data["transl"] -= self.mocap_delta_pos

        # remove root translation
        smplx_data["transl"] = [[0,0,0.9]]
        human_height = 1.7


        smplx_output = self.body_model(
            # betas=torch.tensor(smplx_data["smpl_betas"]).float().view(1, -1), # (16,)
            global_orient=torch.tensor(smplx_data["global_orient_amass"][:1]).float(), # (N, 3)
            body_pose=torch.tensor(smplx_data["body_pose"][:1,:63]).float(), # (N, 63)
            transl=torch.tensor(smplx_data["transl"][:1]).float(), # (N, 3)
            left_hand_pose=torch.zeros(1, 45).float(),
            right_hand_pose=torch.zeros(1, 45).float(),
            jaw_pose=torch.zeros(1, 3).float(),
            leye_pose=torch.zeros(1, 3).float(),
            reye_pose=torch.zeros(1, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        return  smplx_output, human_height
    


    def parse_smplx_output(self, smplx_output, body_model):
        global_orient = smplx_output.global_orient
        full_body_pose = smplx_output.full_pose.reshape(1, -1, 3)
        joints = smplx_output.joints.detach().numpy()
        joint_names = JOINT_NAMES[: len(body_model.parents)]
        parents = body_model.parents


        result = {}
        single_global_orient = global_orient[0]
        single_full_body_pose = full_body_pose[0]
        single_joints = joints[0]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        return result

    def _stepper_loop(self):
        """Run viewer stepping at fixed target FPS using the latest qpos."""
        target_fps = max(1, int(self.config.tgt_fps))
        period = 1.0 / float(target_fps)
        next_ts = time.perf_counter()
        while not self._step_stop.is_set():
            # Lazy-create viewer in this thread to keep GL context/thread affinity
            if self.robot_motion_viewer is None and self.config.viz_retgt:
                try:
                    self.robot_motion_viewer = RobotMotionViewer(
                        robot_type=self.config.robot,
                        motion_fps=self.config.tgt_fps,
                        transparent_robot=0,
                        record_video=self.config.record_video,
                        video_path=self.config.save_path + f"/{self.config.robot}.mp4",
                    )
                except Exception:
                    # If viewer creation fails, retry next tick
                    pass
            start = time.perf_counter()
            # Wait until we have a qpos
            qpos = None
            with self._qpos_lock:
                if self._latest_qpos is not None:
                    qpos = self._latest_qpos.copy()
            if qpos is not None and self.robot_motion_viewer is not None:
                try:
                    self.robot_motion_viewer.step(
                        root_pos=qpos[:3],
                        root_rot=qpos[3:7],
                        dof_pos=qpos[7:],
                        human_motion_data=(self.retgt_module.scaled_human_data if self.retgt_module is not None else None),
                        human_pos_offset=np.array([0.0, 0.0, 0.0]),
                        show_human_body_name=False,
                        rate_limit=False,  # external pacing
                    )

                    if self.config.save_retgt:
                        qpos[3:7] = qpos[3:7][[1,2,3,0]]
                        self.save_retgt_buffer.append(qpos.copy())


                except Exception:
                    # Keep loop alive even if a frame fails
                    pass
            # sleep to maintain rate
            next_ts += period
            delay = next_ts - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                # if we're behind, skip sleep and realign
                next_ts = time.perf_counter()



def smooth_qpos(qpos_seq: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Smooth a sequence of MuJoCo-style qpos vectors while respecting quaternion math.

    Args:
        qpos_seq: (T, D) array. Each row is a qpos where:
                  qpos[0:3]  -> root position (x, y, z)
                  qpos[3:7]  -> root orientation quaternion (w, x, y, z) or (x, y, z, w) if you prefer—set order below
                  qpos[7:]   -> remaining DOF positions
        window_size: odd integer >= 3 specifying the smoothing window.

    Returns:
        (T, D) array of smoothed qpos.
    """
    assert qpos_seq.ndim == 2, "qpos_seq must be (T, D)"
    T, D = qpos_seq.shape
    assert D >= 7, "qpos must have at least 7 elements (pos3 + quat4)"
    if window_size < 3:
        return qpos_seq.copy()
    if window_size % 2 == 0:
        window_size += 1  # make it odd for symmetric window

    # ---- Helpers ----
    def _moving_average(x, w):
        # Pad with edge values to keep length and avoid lag at boundaries
        pad = w // 2
        xpad = np.pad(x, ((pad, pad), (0, 0)), mode='edge') if x.ndim == 2 else np.pad(x, (pad, pad), mode='edge')
        kernel = np.ones(w, dtype=x.dtype) / float(w)
        if x.ndim == 1:
            return np.convolve(xpad, kernel, mode='valid')
        else:
            # Convolve each column independently
            out = np.empty_like(x, dtype=np.float64)
            for i in range(x.shape[1]):
                out[:, i] = np.convolve(xpad[:, i], kernel, mode='valid')
            return out

    def _normalize_quat(q):
        # Normalize last axis
        norm = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
        return q / norm

    def _enforce_quat_sign_continuity(quats):
        """
        Ensures consecutive quaternions lie on the same hemisphere:
        if dot(q_t, q_{t-1}) < 0 then flip q_t.
        """
        qs = quats.copy()
        for t in range(1, qs.shape[0]):
            if np.dot(qs[t], qs[t - 1]) < 0.0:
                qs[t] = -qs[t]
        return qs

    def _smooth_quat_windowed(quats, w):
        """
        Uniform windowed average of quaternions with sign continuity and renorm.
        For small windows and already-smooth data this works well; for highly
        curved orientation trajectories, consider SLERP-based splines.
        """
        pad = w // 2
        qs = _enforce_quat_sign_continuity(_normalize_quat(quats))
        # Pad with edge values (already sign-corrected)
        qpad = np.pad(qs, ((pad, pad), (0, 0)), mode='edge').astype(np.float64)
        out = np.empty_like(qs, dtype=np.float64)
        for t in range(qs.shape[0]):
            qwin = qpad[t:t + w]  # (w, 4)
            # Re-align signs within the window to the center frame to avoid cancellation
            center = qwin[w // 2]
            aligned = qwin.copy()
            dots = (aligned @ center)
            aligned[dots < 0.0] *= -1.0
            qavg = aligned.mean(axis=0)
            out[t] = qavg
        return _normalize_quat(out)

    # ---- Split fields ----
    root_pos = qpos_seq[:, 0:3]
    root_quat = qpos_seq[:, 3:7]
    dof_pos  = qpos_seq[:, 7:] if D > 7 else None

    # ---- Smooth each part ----
    root_pos_s = _moving_average(root_pos, window_size)
    root_quat_s = _smooth_quat_windowed(root_quat, window_size)
    if dof_pos is not None and dof_pos.shape[1] > 0:
        dof_pos_s = _moving_average(dof_pos, window_size)
        qpos_smoothed = np.concatenate([root_pos_s, root_quat_s, dof_pos_s], axis=1)
    else:
        qpos_smoothed = np.concatenate([root_pos_s, root_quat_s], axis=1)

    # Preserve original dtype
    return qpos_smoothed.astype(qpos_seq.dtype)


def test_RetgtModule(args):
    
    mocap_module_cfg = parse_MocapModule_cfg(args)
    mocap_module_class = eval(args.MocapModule)
    mocap_module = mocap_module_class(mocap_module_cfg)

    retgt_module_cfg = parse_RetgtModule_cfg(args)
    retgt_module_class = eval(args.RetgtModule)
    retgt_module = retgt_module_class(retgt_module_cfg)
    

    frame_count = 0
    frame = cv2.imread(ENV_DIR+'/modules/test_images/Switch_input.png')

    while True:
        
        mocap_reuslt = mocap_module(frame)
        retgt_module.retarget(mocap_reuslt)


        if mocap_reuslt is not None:
            if frame_count ==0:
                start_time = time.time()
            frame_count += 1
            if frame_count == 100:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Measured FPS: {100/elapsed_time:.2f}")


        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    
    cv2.destroyAllWindows()
    retgt_module.close()



if __name__ == "__main__":

    
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    test_RetgtModule(args)
    

  
