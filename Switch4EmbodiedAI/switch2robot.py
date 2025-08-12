
import pathlib
import os
import time
import signal

import cv2
import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
from scipy.spatial.transform import Rotation as R

from third_party.GMR.general_motion_retargeting import GeneralMotionRetargeting as GMR
from third_party.GMR.general_motion_retargeting import RobotMotionViewer
from third_party.GMR.general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

from Switch4EmbodiedAI.utils.helpers import get_args
from Switch4EmbodiedAI.utils.helpers import signal_handler, build_modules



class Switch2Robot_Module():
    '''
    Capture motion from Switch Game, retarget to humanoid robot
    '''
    def __init__(self, args):
        self.config = args
        self.stream_module, self.mocap_module, self.retgt_module = build_modules(args)

        self.start_stream = False
        self.start_mocap = False
        self.start_retgt = False



    




def main(args):
    
    



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
    main(args)