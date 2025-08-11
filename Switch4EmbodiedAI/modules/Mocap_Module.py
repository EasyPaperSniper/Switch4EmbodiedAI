import cv2
import time
import torch
from torch import nn
import numpy as np
from romp import romp_settings, ROMP
from vis_human import setup_renderer, rendering_romp_bev_results
from romp.post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from romp.utils import img_preprocess, create_OneEuroFilter, euclidean_distance, check_filter_state, \
    time_cost, download_model, determine_device, convert_cam_to_3d_trans,\
    wait_func, collect_frame_path, progress_bar, get_tracked_ids, smooth_results, convert_tensor2numpy, save_video_results
from Switch4EmbodiedAI import ROOT_DIR, RESULT_DIR, ENV_DIR


class MocapModule(nn.Module):
    def __init__(self, config):
        super().__init__()
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
            cv2.imshow('rendered', outputs['rendered_image'][:,image.shape[1]:])
            wait_func(self.mocap_module.settings.mode)

        return convert_tensor2numpy(outputs)

        
   
def test_MocapModule(args):
    # stream_module_cfg = parse_StreamModule_cfg(args)
    mocap_module_cfg = parse_MocapModule_cfg(args)


    # stream_module_class = eval(args.StreamModule)
    # stream_module = stream_module_class(stream_module_cfg)
    mocap_module_class = eval(args.MocapModule)
    mocap_module = mocap_module_class(mocap_module_cfg)
    
    
    # stream_module.start()

    frame_count = 0
    frame = cv2.imread(ENV_DIR+'modules/test_images/Switch_input.png')

    while True:
        # frame = stream_module.read()
        
        if frame is None:
            break
        
        
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
        if cv2.waitKey(1) & 0xFF == ord('q') :#or stream_module.stopped:
            break

    # stream_module.stop()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    from Switch4EmbodiedAI.utils.helpers import get_args, parse_StreamModule_cfg, parse_MocapModule_cfg
    from Switch4EmbodiedAI.modules.Stream_Module import *
    args = get_args()
    test_MocapModule(args)
    