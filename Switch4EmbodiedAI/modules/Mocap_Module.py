import cv2
import time
import signal
import os
import datetime

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
        self.stopped = False
        self.outputs = None

    def forward(self, file_path):
        # Load mocap data from the specified file path
        pass

    def add_amassFrame(self, output):
        # Process the loaded mocap data
        pass

    def close(self):
        self.stopped = True


    def start(self):
        self.stopped = False



class ROMP_MocapModule(MocapModule):
    def __init__(self, config):
        super().__init__(config)
        self.mocap_module = ROMP(config)
        self.outputs = {}
        self._video_writer = None
        self._video_path = None
        # Measured-FPS recording (default): buffer frames until we can estimate fps, then open writer
        self._buffer_frames = []
        self._buffer_start_time = None
        self._fps_locked = False
        self._measured_fps = None

    def forward(self, image, signal_ID=0, **kwargs):
        outputs, image_pad_info = self.mocap_module.single_image_forward(image)
        self.outputs['rendered_image'] = image
        if outputs is None:
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
        self.outputs = outputs
        self._maybe_write_video()
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
    

    def viz_outputs(self):
        if self.mocap_module.settings.show:
            cv2.imshow('rendered', self.outputs['rendered_image'])


    def _maybe_write_video(self):
        if not getattr(self.config, 'save_video', False):
            return
        if self.outputs is None or 'rendered_image' not in self.outputs:
            return
        frame = self.outputs['rendered_image']
        if frame is None:
            return
        # If writer not yet initialized, buffer frames and measure FPS first
        if self._video_writer is None:
            if self._buffer_start_time is None:
                self._buffer_start_time = time.time()
            self._buffer_frames.append(frame.copy())
            elapsed = time.time() - self._buffer_start_time
            # Lock FPS after at least ~1s or 30 frames, whichever comes first
            if (not self._fps_locked) and (elapsed >= 1.0 or len(self._buffer_frames) >= 30):
                frames = len(self._buffer_frames)
                self._measured_fps = max(1.0, frames / max(elapsed, 1e-6))
                save_dir = self.config.save_path if self.config.save_path is not None else os.getcwd()
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                self._video_path = os.path.join(save_dir, f'romp_{timestamp}.mp4')
                height, width = frame.shape[:2]
                self._video_writer = cv2.VideoWriter(
                    self._video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    float(self._measured_fps),
                    (width, height),
                )
                for bf in self._buffer_frames:
                    self._video_writer.write(bf)
                self._buffer_frames.clear()
                self._fps_locked = True
            return

        # Normal path: writer initialized, write 1:1 frames
        self._video_writer.write(frame)

    def close(self):
        super().close()
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            if self._video_path is not None:
                print(f"ROMP video saved to {self._video_path}")
        self._buffer_frames.clear()
        self._buffer_start_time = None
        self._fps_locked = False
        self._measured_fps = None


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
        mocap_module.viz_outputs()

        if mocap_reuslt is not None:
            if frame_count ==0:
                start_time = time.time()
            frame_count += 1
            if frame_count == 100:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Measured FPS: {100/elapsed_time:.2f}")

        # Exit on 'esc' key press
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
    