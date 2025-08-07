from .configclass import *

class GMR_RetargetingModuleConfig(BaseConfig):
    smplx_file = None  # Path to the SMPLX motion file
    robot = "unitree_g1"  # Choose from ["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"]
    save_path = None  # Path to save the robot motion
    loop = False  # Whether to loop the motion
    record_video = False  # Whether to record the video
    rate_limit = False  # Whether to limit the rate of the retargeted robot motion to keep the same as the human motion


