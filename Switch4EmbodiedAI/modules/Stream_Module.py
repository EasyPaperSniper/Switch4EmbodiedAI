from threading import Thread

import cv2
import torch
import numpy as np


class SimpleStreamModule:
    '''
    Get image from capture card.
    No complex processing, just a placeholder for future image processing tasks.
    '''
    def __init__(self, config):
        self.config = config
        self.capture_card_index = config.capture_card_index
        self.stream = cv2.VideoCapture(self.capture_card_index)

        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False


    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        Thread(target=self.monitor_input, daemon=True).start()
        return self
    

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()


    def monitor_input(self):
        while not self.stopped:
            user_input = input()
            if user_input.strip().lower() == 'q':
                self.stop()
                return
            
            

    def read(self):
        # return the frame most recently read
        return self.frame
    

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # if cv2.waitKey(1)  == ord('q'):
        #     break

    def save_frame(self, frame, path):
        if frame:
            cv2.imwrite(path, frame)
            return True
        return False


    def process_frame(self, frame):
        return frame
    
    



def test_StreamModule(stream_module_cfg):

    Stream_module = SimpleStreamModule(stream_module_cfg)
    Stream_module.start()

    while True:
        frame = Stream_module.read()
        if frame is None:
            break
        

        if stream_module_cfg.viz_stream:
            cv2.imshow("Stream Module Output", frame)


        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q') or Stream_module.stopped:
            Stream_module.stop()
            break


    
    cv2.destroyAllWindows()




if __name__ == '__main__':
    
    
    from Switch4EmbodiedAI.utils.helpers import get_args, parse_StreamModule_cfg
    args = get_args()
    stream_module_cfg = parse_StreamModule_cfg(args)

    test_StreamModule(stream_module_cfg)