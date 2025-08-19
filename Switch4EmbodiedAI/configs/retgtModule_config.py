import os.path as osp
from .configclass import *
from Switch4EmbodiedAI import ROOT_DIR, RESULT_DIR, ENV_DIR

class GMR_RetgtModuleConfig(BaseConfig):
    smplx_file = None  # Path to the SMPLX motion file
    robot = "unitree_g1"  # Choose from ["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"]
    smplx_file = osp.join(ENV_DIR, 'utils', 'smpl_model_data','SMPLX_NEUTRAL.npz') # Folder containing SMPLX model files
    viz_retgt = True  # Whether to visualize the retargeted motion
    save_retgt = True  # Whether to save the retargeted motion
    save_path = None  # Path to save the robot motion
    record_video = True  # Whether to record the video
    rate_limit = True  # Whether to limit the rate of the retargeted robot motion to keep the same as the human motion
    tgt_fps = 30  # Target frames per second for the retargeted motion

