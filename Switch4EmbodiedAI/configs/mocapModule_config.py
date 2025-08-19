import os.path as osp
from Switch4EmbodiedAI import ROOT_DIR, RESULT_DIR, ENV_DIR
from .configclass import *



class ROMP_MocapModuleConfig(BaseConfig):
    mode = 'webcam'  # 'image', 'video', 'webcam'
    input = None  # Path to the input image / video
    save_path = None #'Path to save the results'
    GPU = 0 # The gpu device number to run the inference on. If GPU=-1, then running in cpu mode
    onnx = False    # Whether to use ONNX for acceleration.
    temporal_optimize = False   # Whether to use OneEuro filter to smooth the results
    center_thresh = 0.25    # The confidence threshold of positive detection in 2D human body center heatmap.
    show_largest = True # Whether to show the largest person only
    smooth_coeff = 3.0  # The smoothness coeff of OneEuro filter, the smaller, the smoother.
    calc_smpl = True  # Whether to calculate the smpl mesh from estimated SMPL parameters
    render_mesh = True # Whether to render the estimated 3D mesh mesh to image
    renderer = 'sim3dr' # Choose the renderer for visualizaiton: pyrender (great but slow), sim3dr (fine but fast)
    show = False # Whether to show the rendered results
    show_items = 'mesh' # The items to visualized, including mesh,pj2d,j3d,mesh_bird_view,mesh_side_view,center_conf. splited with ,
    save_video = True # Whether to save the video results
    frame_rate = 30 # The frame_rate of saved video results
    smpl_path = osp.join(ENV_DIR, 'utils', 'romp','SMPL_NEUTRAL.pth') # The path of smpl model file
    model_path = osp.join(ENV_DIR, 'utils', 'romp', 'ROMP.pkl') # The path of ROMP checkpoint
    model_onnx_path = osp.join(ENV_DIR, 'utils', 'romp', 'ROMP.onnx') # The path of ROMP onnx checkpoint
    root_align = False # Please set this config as True to use the ROMP checkpoints trained by yourself.
    webcam_id = 0 # The Webcam ID.


