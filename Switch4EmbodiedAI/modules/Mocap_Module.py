import cv2
import torch
import numpy as np


class MocapModule:
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
        self.romp = ROMP(romp_settings)

    def load_mocap_data(self, file_path):
        # Load ROMP mocap data from the specified file path
        self.mocap_data = self.romp.load(file_path)

    def process_mocap_data(self):
        # Process the loaded ROMP mocap data
        self.mocap_data = self.romp.process(self.mocap_data)

    def save_mocap_data(self, save_path):
        # Save the processed mocap data to the specified path
        if self.mocap_data is not None:
            torch.save(self.mocap_data, save_path)
        else:
            raise ValueError("No mocap data to save.")
        

if __name__ == "__main__":
    # load image and give output