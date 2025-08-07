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
        self.mocap_module = ROMP(romp_settings)

        
    def forward(self, image, signal_ID=0, **kwargs):
        outputs, image_pad_info = self.mocap_module.single_image_forward(image)
        if outputs is None:
            return None
        if self.settings.temporal_optimize:
            outputs = self.mocap_module.temporal_optimization(outputs, signal_ID)
        outputs['cam_trans'] = self.mocap_module.convert_cam_to_3d_trans(outputs['cam'])
        if self.settings.calc_smpl:
            outputs = self.smpl_parser(outputs, root_align=self.settings.root_align) 
            outputs.update(body_mesh_projection2image(outputs['joints'], outputs['cam'], vertices=outputs['verts'], input2org_offsets=image_pad_info))
        if self.settings.render_mesh:
            rendering_cfgs = {'mesh_color':'identity', 'items': self.visualize_items, 'renderer': self.settings.renderer} # 'identity'
            outputs = rendering_romp_bev_results(self.renderer, outputs, image, rendering_cfgs)
        if self.settings.show:
            cv2.imshow('rendered', outputs['rendered_image'])
            wait_func(self.settings.mode)
        return convert_tensor2numpy(outputs)
        return self.mocap_module(image, signal_ID=signal_ID, **kwargs)
    




if __name__ == "__main__":
    # load image and give output