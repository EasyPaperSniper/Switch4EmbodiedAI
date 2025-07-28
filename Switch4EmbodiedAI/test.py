import cv2
import torch
import argparse
from romp import romp_settings, ROMP
from romp.utils import WebcamVideoStream




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
        # process the frame if needed, such as masking, resizing, etc.
        cv2.imshow('Nintendo Switch Stream', frame)


        if cv2.waitKey(1)  == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    capture_card_index = 0  # Set your capture card index here


    romp_args = romp_settings(['--mode', 'webcam', 
                               '--webcam_id', str(capture_card_index), 
                               '--smpl_path', './Switch4EmbodiedAI/utils/romp/SMPL_NEUTRAL.pth',
                               '--model_path','./Switch4EmbodiedAI/utils/romp/ROMP.pkl',
                               '--model_onnx_path','./Switch4EmbodiedAI/utils/romp/ROMP.onnx',
                               '--show'])

    romp = ROMP(romp_args)

    main(romp_args, romp)