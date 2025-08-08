import cv2
import torch
from torch import nn
import numpy as np
from romp import romp_settings, ROMP


class MocapModule(nn.Module):
    def __init__(self, config):
        self.config = config
        self.mocap_data = None

    def load_mocap_data(self, file_path):
        # Load mocap data from the specified file path
        pass

    def process_mocap_data(self):
        # Process the loaded mocap data
        pass

    def get_mocap_data(self):
        return self.mocap_data
    


class ROMP_MocapModule(MocapModule):
    def __init__(self, config):
        super().__init__(config)
        self.mocap_module = ROMP(config)

        
   
def test_MocapModule(args):
    stream_module_cfg = parse_StreamModule_cfg(args)
    mocap_module_cfg = parse_MocapModule_cfg(args)


    stream_module_class = eval(args.StreamModule)
    stream_module = stream_module_class(stream_module_cfg)
    mocap_module_class = eval(args.MocapModule)
    mocap_module = mocap_module_class(mocap_module_cfg)
    
    
    stream_module.start()

    while True:
        frame = stream_module.read()
        if frame is None:
            break
        
        
        # Process the frame (e.g., display it)
        if stream_module_cfg.viz_stream:
            cv2.imshow("Stream Module Output", frame)


        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q') or stream_module.stopped:
            stream_module.stop()
            break

    
    cv2.destroyAllWindows()




if __name__ == "__main__":
    from Switch4EmbodiedAI.utils.helpers import get_args, parse_StreamModule_cfg, parse_MocapModule_cfg
    from Switch4EmbodiedAI.modules.Stream_Module import *
    args = get_args()
    test_MocapModule(args)
    