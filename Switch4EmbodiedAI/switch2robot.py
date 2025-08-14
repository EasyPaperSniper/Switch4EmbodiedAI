
import pathlib
import os
import time
import signal
from threading import Thread

import cv2
import numpy as np

from Switch4EmbodiedAI.modules import *
from Switch4EmbodiedAI.utils.helpers import *



class Switch2Robot_Module():
    '''
    Capture motion from Switch Game, retarget to humanoid robot
    '''
    def __init__(self, args):
        self.config = args
        self.stream_module, self.mocap_module, self.retgt_module = self.build_modules(args)


        self.run_mocap = False
        self.run_retgt = False
        self.stopped  = False


    def build_modules(self,args):
        stream_module_cfg = parse_StreamModule_cfg(args)
        stream_module_class = eval(args.StreamModule)
        stream_module = stream_module_class(stream_module_cfg)

        mocap_module_cfg = parse_MocapModule_cfg(args)
        mocap_module_class = eval(args.MocapModule)
        mocap_module = mocap_module_class(mocap_module_cfg)

        retgt_module_cfg = parse_RetgtModule_cfg(args)
        retgt_module_class = eval(args.RetgtModule)
        retgt_module = retgt_module_class(retgt_module_cfg)

        return stream_module, mocap_module, retgt_module


    def start(self,):  
        self.userCtrl_thread = Thread(target=self.user_input, args=(), daemon=True)
        self.userCtrl_thread.start()
        self.stream_module.start()


    def view_Switch(self,):
        if (self.run_mocap) and self.mocap_module.mocap_module.settings.show and (self.mocap_module.outputs is not None):
            self.mocap_module.viz_outputs()
        elif  self.stream_module.config.viz_stream:
            self.stream_module.viz_frame()
        cv2.waitKey(1)

        


    def close(self):
        self.stopped = True
        self.retgt_module.close()
        self.stream_module.close()
        cv2.destroyAllWindows()
        self.userCtrl_thread.join()
        self.switchView_thread.join()



    def forward(self):
        stream_frame, mocap_output, retgt_output = None, None, None
        stream_frame = self.stream_module.read()
        if self.run_mocap:
            mocap_output = self.mocap_module(stream_frame)
            
        if self.retgt_module.robot_motion_viewer is None:
            self.view_Switch()
        if self.run_retgt and mocap_output:
            retgt_output = self.retgt_module.retarget(mocap_output)
        
            
        return stream_frame, mocap_output, retgt_output



    def user_input(self):
        while True:
            user_input = input()
            if user_input.strip().lower() == 'q':
                print('========= Quit Program ==========')
                self.close()
                return
            if user_input.strip().lower() == 'a':
                print('========= Start Viz Streaming ==========')
                self.stream_module.config.viz_stream = True
                self.config.viz_stream = True
            if user_input.strip().lower() == 'w':
                print('========= Start Motion Trackng ==========')
                cv2.destroyAllWindows()
                self.run_mocap = True
                self.config.viz_mocap = True
                self.mocap_module.mocap_module.settings.show = True
            if user_input.strip().lower() == 'e':
                print('========= Start Motion Retargeting ==========')
                cv2.destroyAllWindows()
                self.run_mocap = True
                self.run_retgt = True
            if user_input.strip().lower() == 's':
                print('========= Stop Motion Trackng ==========')
                self.mocap_module.mocap_module.settings.show = False
                self.config.viz_mocap = False
                self.run_mocap = False
                self.run_retgt = False
                cv2.destroyAllWindows()
            if user_input.strip().lower() == 'd':
                print('========= Stop Motion Retargeting ==========')
                self.run_retgt = False
                self.retgt_module.close()



def test_switch2RobotModule(args):
    Switch2Robot = Switch2Robot_Module(args)
    Switch2Robot.start()

    while not Switch2Robot.stopped:
        stream_frame, mocap_output, retgt_output = Switch2Robot.forward()


    
    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    test_switch2RobotModule(args)