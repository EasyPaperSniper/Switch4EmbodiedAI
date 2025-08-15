import signal
import numpy as np



from Switch4EmbodiedAI.utils.helpers import *
from Switch4EmbodiedAI.switch2robot import Switch2Robot_Module




def test_GMT(args):
    args.viz_retgt = False
    Switch2Robot = Switch2Robot_Module(args)
    Switch2Robot.start()


    while not Switch2Robot.stopped:
        stream_frame, mocap_output, retgt_output = Switch2Robot.forward()
        


    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    test_GMT(args)