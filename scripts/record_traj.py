import signal
import pickle
import numpy as np
import datetime

from Switch4EmbodiedAI.utils.helpers import *
from Switch4EmbodiedAI.switch2robot import Switch2Robot_Module




def save_retgt_traj(args):
    Switch2Robot = Switch2Robot_Module(args)
    Switch2Robot.start()


    save_retgt_traj = None
    while not Switch2Robot.stopped:
        stream_frame, mocap_output, retgt_output = Switch2Robot.forward()
        if Switch2Robot.run_retgt and (retgt_output is not None):
            if save_retgt_traj is None:
                save_retgt_traj = []
                now = datetime.datetime.now()
                save_path = args.save_path + now.strftime("%m-%d_%H-%M.pkl")

            save_retgt_traj.append(retgt_output)
        elif not save_retgt_traj:
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
                pickle.dump(motion_data, f)
            print(f"Saved to {save_path}")
            save_retgt_traj = None


    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    save_retgt_traj(args)