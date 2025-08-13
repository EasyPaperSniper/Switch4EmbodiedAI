import argparse
import pathlib
import os
import time
import signal

import numpy as np
import smplx
from smplx.joint_names import JOINT_NAMES
from scipy.spatial.transform import Rotation as R

from third_party.GMR.general_motion_retargeting import GeneralMotionRetargeting as GMR
from third_party.GMR.general_motion_retargeting import RobotMotionViewer
from third_party.GMR.general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

from Switch4EmbodiedAI.utils.helpers import get_args, parse_StreamModule_cfg, parse_MocapModule_cfg, parse_RetgtModule_cfg
from Switch4EmbodiedAI.modules.Mocap_Module import *
from Switch4EmbodiedAI.utils.helpers import signal_handler



class GMR_RetgtModule():
    def __init__(self, config):
        
        self.config = config
        self.retgt_module = None
        self.robot_motion_viewer = None


    def reset(self):
        self.retgt_module = None
        self.robot_motion_viewer = None



    def init_module(self, actual_human_height):
        if self.retgt_module is None:
            self.retgt_module = GMR(
                src_human="smplx",
                tgt_robot=self.config.robot,
                actual_human_height=actual_human_height,
                )
            if self.config.viz_retgt:
                self.robot_motion_viewer = RobotMotionViewer(robot_type=self.config.robot,
                                                    motion_fps=self.config.tgt_fps,
                                                    transparent_robot=0,
                                                    record_video=self.config.record_video,
                                                    video_path=self.config.save_path + f"/{self.config.robot}.mp4",)

        

    def retarget(self, input_data):
        # retarget the input data to the robot motion
        smplx_data, body_model, smplx_output, actual_human_height = self.load_smplx_file(
            input_data, self.config.smplx_file)
        

        smplx_data_frames = self.parse_smplx_output(smplx_output ,body_model)
        self.init_module(actual_human_height)
        qpos = self.retgt_module.retarget(smplx_data_frames)



        if self.config.viz_retgt:
            self.robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=self.retgt_module.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=self.config.rate_limit,
            )
        
        return qpos
    
    def close(self):
        """Close the robot motion viewer if it exists."""
        if self.robot_motion_viewer is not None:
            self.robot_motion_viewer.close()
            print("Robot motion viewer closed.")
        self.reset()


    

    def load_smplx_file(self, smplx_data, smplx_body_model_path):
        body_model = smplx.create(
            smplx_body_model_path,
            "smplx",
            gender="NEUTRAL",
            use_pca=False,
        )
    
        num_frames = smplx_data["body_pose"].shape[0]
        if len(smplx_data["smpl_betas"].shape)==1:
            human_height = 1.66 + 0.1 * smplx_data["smpl_betas"][0]
        else:
            human_height = 1.66 + 0.1 * smplx_data["smpl_betas"][0, 0]
        
        # adjust the camera height
        if self.retgt_module is None:
            self.mocap_delta_pos = smplx_data["transl"][0].copy()
            self.mocap_delta_pos[2] = smplx_data["transl"][0,2] - human_height * 0.5 -0.05

        smplx_data["transl"] -= self.mocap_delta_pos

        smplx_output = body_model(
            betas=torch.tensor(smplx_data["smpl_betas"]).float().view(1, -1), # (16,)
            global_orient=torch.tensor(smplx_data["global_orient_amass"]).float(), # (N, 3)
            body_pose=torch.tensor(smplx_data["body_pose"][:,:63]).float(), # (N, 63)
            transl=torch.tensor(smplx_data["transl"]).float(), # (N, 3)
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        
        return smplx_data, body_model, smplx_output, human_height
    

    def parse_smplx_output(self, smplx_output, body_model):
        global_orient = smplx_output.global_orient
        full_body_pose = smplx_output.full_pose.reshape(1, -1, 3)
        joints = smplx_output.joints.detach().numpy()
        joint_names = JOINT_NAMES[: len(body_model.parents)]
        parents = body_model.parents


        result = {}
        single_global_orient = global_orient[0]
        single_full_body_pose = full_body_pose[0]
        single_joints = joints[0]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        return result



def test_RetgtModule(args):
    
    mocap_module_cfg = parse_MocapModule_cfg(args)
    mocap_module_class = eval(args.MocapModule)
    mocap_module = mocap_module_class(mocap_module_cfg)

    retgt_module_cfg = parse_RetgtModule_cfg(args)
    retgt_module_class = eval(args.RetgtModule)
    retgt_module = retgt_module_class(retgt_module_cfg)
    

    frame_count = 0
    frame = cv2.imread(ENV_DIR+'/modules/test_images/Switch_input.png')

    while True:
        
        mocap_reuslt = mocap_module(frame)
        retgt_module.retarget(mocap_reuslt)


        if mocap_reuslt is not None:
            if frame_count ==0:
                start_time = time.time()
            frame_count += 1
            if frame_count == 100:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Measured FPS: {100/elapsed_time:.2f}")


        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    
    cv2.destroyAllWindows()
    retgt_module.close()



if __name__ == "__main__":

    
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals
    args = get_args()
    test_RetgtModule(args)
    

  
