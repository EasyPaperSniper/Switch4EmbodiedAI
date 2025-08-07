from threading import Thread

import cv2
import torch
import numpy as np


class SimpleImageModule:
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
    
    



def test_imageModule(args):

    image_module = SimpleImageModule(args)
    image_module.start()

    while True:
        frame = image_module.read()
        if frame is None:
            break
        
        # Process the frame (e.g., display it)
        cv2.imshow("Capture Card Frame", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    image_module.stop()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    
    
    import argparse
    parser = argparse.ArgumentParser(description="Image Module Test")
    parser.add_argument("--capture_card_index", type=int, default=0, help="Index of the capture card.")
    args = parser.parse_args()


    test_imageModule(args)