import os
import os.path as osp
import sys
import argparse

import torch
import cv2

from Switch4EmbodiedAI import ROOT_DIR, RESULT_DIR, ENV_DIR
from Switch4EmbodiedAI.configs import *
from romp.utils  import download_model




def get_args():
    parser = argparse.ArgumentParser(description="Switch4EAI")
    
    # group general
    parser.add_argument("--device", type=int, default=-1, help="GPU id to use, -1 for CPU.")
    parser.add_argument("--capture_card_index", type=int, default=None, help="Index of the capture card.")
    parser.add_argument("--save_path", type=str, default=RESULT_DIR, help = 'Path to save the results')



    # group stream Module
    stream_group = parser.add_argument_group("StreamModule", description="Arguments for stream Module Setting.")
    stream_group.add_argument('--StreamModule', type=str, default='SimpleStreamModule', help='The stream module to use.')
    stream_group.add_argument('--viz_stream', action='store_true', help='Whether to visualize the stream module output.')
    stream_group.add_argument('--save_stream', action='store_true', help='Whether to save the stream module output.')



    # group Mocap Module
    mocap_group = parser.add_argument_group("Mocap", description="Arguments for Mocap Setting.")
    mocap_group.add_argument('--MocapModule', type=str, default='ROMP_MocapModule', help='The mocap module to use.')
    mocap_group.add_argument('--viz_mocap', action='store_true', help='Whether to visualize the mocap module output.')
    mocap_group.add_argument('--save_mocap', action='store_true', help='Whether to save the mocap module output.')
   

    # group motion retageting
    retargeting_group = parser.add_argument_group("Retargeting", description="Arguments for Retargeting Setting.")
    retargeting_group.add_argument('--RetgtModule', type=str, default='GMR_RetgtModule', help='The retargeting module to use.')
    retargeting_group.add_argument('--robot', type=str, default='unitree_g1', help='The robot to use for retargeting.')
    retargeting_group.add_argument('--viz_retgt', action='store_false', help='Whether to visualize the retargeted motion.')
    retargeting_group.add_argument('--save_retgt', action='store_true', help='Whether to save the retargeted motion.')
    retargeting_group.add_argument('--record_retgt', action='store_true', help='Whether to record the video of the retargeted motion.')


    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = -1

    
    # if args.viz_retgt:
    #     args.viz_mocap = False
    #     args.viz_stream = False
    # if args.viz_mocap:
    #     args.viz_stream = False
    return args




def parse_StreamModule_cfg(args):

    StreamModule_cfg =eval(args.StreamModule+'Config()')
    if args.capture_card_index is not None:
       StreamModule_cfg.capture_card_index = args.capture_card_index
    StreamModule_cfg.viz_stream = args.viz_stream
    StreamModule_cfg.save_stream = args.save_stream
    StreamModule_cfg.save_path = args.save_path

    return StreamModule_cfg




def parse_MocapModule_cfg(args):

    mocap_cfg = eval(args.MocapModule+'Config()')
    mocap_cfg.GPU = args.device
    if args.device == -1:
        mocap_cfg.temporal_optimize = False
    mocap_cfg.save_path = args.save_path

    if args.viz_mocap:
        mocap_cfg.show = True
    if mocap_cfg.show:
        mocap_cfg.render_mesh = True
    if  mocap_cfg.render_mesh or mocap_cfg.show_largest:
        mocap_cfg.calc_smpl = True
    if args.save_mocap:
        mocap_cfg.save_video = True


    if not os.path.exists(mocap_cfg.smpl_path):
        if os.path.exists(mocap_cfg.smpl_path.replace('SMPL_NEUTRAL.pth', 'smpl_packed_info.pth')):
            mocap_cfg.smpl_path = mocap_cfg.smpl_path.replace('SMPL_NEUTRAL.pth', 'smpl_packed_info.pth')
        print('please prepare SMPL model files following instructions at https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation')
    if not os.path.exists(mocap_cfg.model_path):
        romp_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl'
        download_model(romp_url, mocap_cfg.model_path, 'ROMP')
    if not os.path.exists(mocap_cfg.model_onnx_path) and mocap_cfg.onnx:
        romp_onnx_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.onnx'
        download_model(romp_onnx_url, mocap_cfg.model_onnx_path, 'ROMP')


    if args.MocapModule == 'ROMP_MocapModule': 
        mocap_cfg.webcam_id = args.capture_card_index 



    return mocap_cfg
    



def parse_RetgtModule_cfg(args):
    retgt_cfg = eval(args.RetgtModule+'Config()')
    if args.robot in ["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"]:
        retgt_cfg.robot = args.robot
    if args.save_retgt:       
        retgt_cfg.save_retgt = True
        retgt_cfg.record_video = args.record_retgt
    else:
        retgt_cfg.record_video = False
    retgt_cfg.viz_retgt = args.viz_retgt
    retgt_cfg.save_path = args.save_path
    

    return retgt_cfg





def signal_handler(sig, frame):
    """Handle termination signals (e.g., Ctrl+C)."""
    print("Terminating program...")
    if 'retgt_module' in globals() and retgt_module is not None:
        retgt_module.close()
    cv2.destroyAllWindows()
    sys.exit(0)