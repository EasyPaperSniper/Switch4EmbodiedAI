import cv2
import time
import signal

import torch
from torch import nn
import numpy as np
from scipy.spatial.transform import Rotation as R

from romp import romp_settings, ROMP
from vis_human import setup_renderer, rendering_romp_bev_results
from romp.post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from romp.utils import img_preprocess, create_OneEuroFilter, euclidean_distance, check_filter_state, \
    time_cost, download_model, determine_device, convert_cam_to_3d_trans,\
    wait_func, collect_frame_path, progress_bar, get_tracked_ids, smooth_results, convert_tensor2numpy, save_video_results
from Switch4EmbodiedAI import ROOT_DIR, RESULT_DIR, ENV_DIR
from Switch4EmbodiedAI.utils.helpers import signal_handler



class MocapModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mocap_data = None

    def forward(self, file_path):
        # Load mocap data from the specified file path
        pass

    def padd_amassFrame(self, output):
        # Process the loaded mocap data
        pass



class ROMP_MocapModule(MocapModule):
    def __init__(self, config):
        super().__init__(config)
        self.mocap_module = ROMP(config)

    def forward(self, image, signal_ID=0, **kwargs):
        outputs, image_pad_info = self.mocap_module.single_image_forward(image)
        if outputs is None:
            if self.mocap_module.settings.show:
                cv2.imshow('rendered', image)
            return None
        if self.mocap_module.settings.temporal_optimize:
            outputs = self.mocap_module.temporal_optimization(outputs, signal_ID)
        outputs['cam_trans'] = convert_cam_to_3d_trans(outputs['cam'])
        if self.mocap_module.settings.calc_smpl:
            outputs = self.mocap_module.smpl_parser(outputs, root_align=self.mocap_module.settings.root_align) 
            outputs.update(body_mesh_projection2image(outputs['joints'], outputs['cam'], vertices=outputs['verts'], input2org_offsets=image_pad_info))
        if self.mocap_module.settings.render_mesh:
            rendering_cfgs = {'mesh_color':'identity', 'items': self.mocap_module.visualize_items, 'renderer': self.mocap_module.settings.renderer} # 'identity'
            outputs = rendering_romp_bev_results(self.mocap_module.renderer, outputs, image, rendering_cfgs)
        if self.mocap_module.settings.show:
            cv2.imshow('rendered', outputs['rendered_image'][:,])
            wait_func(self.mocap_module.settings.mode)

        return self.add_amassFrame(convert_tensor2numpy(outputs))
    


    def add_amassFrame(self, outputs):
        '''
        Add AMASS coordinate as part of the outputs
        '''
        # Convert the outputs to AMASS format
        global_orient, cam_trans = outputs['global_orient'],  outputs['cam_trans']


        # Flip the Y-axis to match AMASS coordinate system
        cam_trans_amass = cam_trans.copy()
        cam_trans_amass[:,2] = cam_trans[:,1]*-1 
        cam_trans_amass[:,1] = cam_trans[:,2]

        # Convert global orientation to AMASS coordinate system
        global_orient_amass = []
        flip_x_axis = R.from_euler('x', -90, degrees=True).as_matrix()
        # TODO: remove for loop do matrix operation
        for i in range(global_orient.shape[0]):
            orient = global_orient[i]
            orient_matrix = R.from_rotvec(orient).as_matrix()
            orient_amass = flip_x_axis @ orient_matrix 
            global_orient_amass.append(R.from_matrix(orient_amass).as_rotvec())
        global_orient_amass = np.array(global_orient_amass)


        outputs['global_orient_amass'] = global_orient_amass
        outputs['transl'] = cam_trans_amass
        return outputs
    
    # TODO: add mirror version for facing monitor play


        
   
def test_MocapModule(args):
    mocap_module_cfg = parse_MocapModule_cfg(args)
    mocap_module_class = eval(args.MocapModule)
    mocap_module = mocap_module_class(mocap_module_cfg)
    
    
    # stream_module.start()

    frame_count = 0
    frame = cv2.imread(ENV_DIR+'/modules/test_images/Switch_input.png')

    while True:        
        mocap_reuslt = mocap_module(frame)

        if mocap_reuslt is not None:
            if frame_count ==0:
                start_time = time.time()
            frame_count += 1
            if frame_count == 100:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Measured FPS: {100/elapsed_time:.2f}")

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == 27: #or stream_module.stopped:
            break

    cv2.destroyAllWindows()





if __name__ == "__main__":
    from Switch4EmbodiedAI.utils.helpers import get_args, parse_StreamModule_cfg, parse_MocapModule_cfg
    from Switch4EmbodiedAI.modules.Stream_Module import *

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    test_MocapModule(args)
    