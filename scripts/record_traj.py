import signal
import pickle
import numpy as np
import datetime
import time

from Switch4EmbodiedAI.utils.helpers import *
from Switch4EmbodiedAI.switch2robot import Switch2Robot_Module




def save_retgt_traj(args):
    Switch2Robot = Switch2Robot_Module(args)
    Switch2Robot.start()

    Switch2Robot.run_mocap = True
    Switch2Robot.run_retgt = True

    # Fixed-rate recording at 30 Hz using the latest qpos (decoupled from ROMP rate)
    target_fps = 30
    period = 1.0 / float(target_fps)
    next_ts = time.perf_counter()

    save_retgt_traj = None
    save_path = None

    while not Switch2Robot.stopped:
        now = time.perf_counter()
        if now < next_ts:
            time.sleep(next_ts - now)
        next_ts += period

        stream_frame, mocap_output, retgt_output = Switch2Robot.forward()

        if not Switch2Robot.run_retgt:
            if save_retgt_traj:
                root_pos = np.array([qpos[:3] for qpos in save_retgt_traj])
                # save from wxyz to xyzw
                root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in save_retgt_traj])
                dof_pos = np.array([qpos[7:] for qpos in save_retgt_traj])

                motion_data = {
                    "root_pos": root_pos,
                    "root_rot": root_rot,
                    "dof_pos": dof_pos,
                }
                with open(save_path, "wb") as f:
                    np.savez(f, **motion_data)
                print(f"Saved to {save_path}, {len(save_retgt_traj)} frames")
            save_retgt_traj = None
            save_path = None
            continue

        # Ensure path and buffer are initialized when recording starts
        if save_retgt_traj is None:
            save_retgt_traj = []
            now_dt = datetime.datetime.now()
            save_path = args.save_path + now_dt.strftime("%m-%d_%H-%M.pkl")
        
        # Acquire qpos at fixed rate: prefer current retgt_output, then latest from retgt module, else duplicate last
        sample_qpos = None
        if retgt_output is not None:
            sample_qpos = retgt_output
        else:
            latest = getattr(Switch2Robot.retgt_module, "_latest_qpos", None)
            if latest is not None:
                sample_qpos = latest

        if sample_qpos is not None:
            save_retgt_traj.append(sample_qpos.copy())
        elif save_retgt_traj:
            # duplicate last to maintain fixed rate
            save_retgt_traj.append(save_retgt_traj[-1].copy())
        else:
            # nothing to record yet this tick
            continue


    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    save_retgt_traj(args)