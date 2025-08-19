import signal
import pickle
import numpy as np
import datetime

from Switch4EmbodiedAI.utils.helpers import *
from Switch4EmbodiedAI.switch2robot import Switch2Robot_Module

from Switch4EmbodiedAI.utils.UDPcomm import UDPComm
# from Switch4EmbodiedAI.utils import UDPComm


def send_retgt_traj(args):

    # UDP setting
    ip = "127.0.0.1"
    send_port = 54010
    client = UDPComm(ip, send_port)


    Switch2Robot = Switch2Robot_Module(args)
    Switch2Robot.start()

    Switch2Robot.run_mocap = True
    Switch2Robot.run_retgt = True

    # save_retgt_traj = None
    while not Switch2Robot.stopped:
        stream_frame, mocap_output, retgt_output = Switch2Robot.forward()

        if Switch2Robot.run_mocap and Switch2Robot.run_retgt:
            # message = {}
            if retgt_output is not None:
                # message["root_pos"] = retgt_output[:3]
                # message["root_rot"] = retgt_output[3:7][[1,2,3,0]] # xyzw
                # message["dof_pos"] = retgt_output[7:]
                retgt_output[3:7] = retgt_output[3:7][[1,2,3,0]]
                client.send_message(retgt_output, ip, send_port)
            else:
                print("no retarget output")
        


    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    send_retgt_traj(args)