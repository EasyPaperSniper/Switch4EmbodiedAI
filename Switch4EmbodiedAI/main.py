import os

import cv2
import torch
import argparse
from romp import romp_settings, ROMP
from romp.utils import WebcamVideoStream





class Switch2SMPL:
    # add args processing and initialization
    # add video strearm processing


    def __init__(self, args=None, **kwargs):
        self.args = args
        self.romp_args = self.parse_args(args)
        self.romp = ROMP(self.romp_args)

        self.cap = WebcamVideoStream(romp_args.webcam_id)




    def process_frame(self, frame):
        outputs = self.romp(frame)
    

    def write_output(self, outputs, output_path):
        # Implement the logic to save the outputs to a file
        pass


    def parse_args(self,  args=None):
        romp_args = romp_settings(['--mode', 'webcam', 
                               '--webcam_id', '0', 
                               '--smpl_path', './Switch4EmbodiedAI/utils/romp/SMPL_NEUTRAL.pth',
                               '--model_path','./Switch4EmbodiedAI/utils/romp/ROMP.pkl',
                               '--model_onnx_path','./Switch4EmbodiedAI/utils/romp/ROMP.onnx',
                               '--calc_smpl',
                               '--show'])



        if not torch.cuda.is_available():
            args.GPU = -1
            args.temporal_optimize = False
        if args.show:
            args.render_mesh = True
        if args.render_mesh or args.show_largest:
            args.calc_smpl = True
        if not os.path.exists(args.smpl_path):
            if os.path.exists(args.smpl_path.replace('SMPL_NEUTRAL.pth', 'smpl_packed_info.pth')):
                args.smpl_path = args.smpl_path.replace('SMPL_NEUTRAL.pth', 'smpl_packed_info.pth')
            print('please prepare SMPL model files following instructions at https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation')
        if not os.path.exists(args.model_path):
            romp_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl'
            download_model(romp_url, args.model_path, 'ROMP')
        if not os.path.exists(args.model_onnx_path) and args.onnx:
            romp_onnx_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.onnx'
            download_model(romp_onnx_url, args.model_onnx_path, 'ROMP')
        return romp_args
    




def main(romp_args, romp):
    cap = cv2.VideoCapture(capture_card_index)  

    # cap = WebcamVideoStream(romp_args.webcam_id)
    # cap.start()
    # while True:
    #     frame = cap.read()
    #     outputs = romp(frame)
    # cap.stop()

    if not cap.isOpened():
        print("Cannot open capture card")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        outputs = romp(frame)
        print(outputs.keys())
        # process the frame if needed, such as masking, resizing, etc.
        cv2.imshow('Nintendo Switch Stream', frame)


        if cv2.waitKey(1)  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    cap.start()
    while True:
        frame = cap.read()
        outputs = romp(frame)
    cap.stop()



if __name__ == '__main__':
    args = get_args()
    agent = Switch2SMPL(args=args)
    main(agent)