
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


class HyperResolutionDataLoader(Dataset):
    def __init__(self, image_root_path, image_source_folder,image_target_folder):
        self.root_path = image_root_path
        def getImageLabels(image_number:int,source_path,target_path):
            source_labels = []
            target_labels = []
            labels = []
            for i in range(image_number):
                sample_id = "{:04d}".format(i + 1)
                source_file_name = sample_id + ".png.tiff"
                target_file_name = sample_id + ".png.tiff"
                source_file_path = os.path.join(source_path, source_file_name)
                target_file_path = os.path.join(target_path, target_file_name)
                source_labels.append(source_file_path)
                target_labels.append(target_file_path)
            labels.append(source_labels)
            labels.append(target_labels)
            return labels
        self.source_path = os.path.join(self.root_path,image_source_folder)
        self.target_path = os.path.join(self.root_path,image_target_folder)
        self.source_files = os.listdir(self.source_path)
        self.target_files = os.listdir(self.target_path)
        self.source_image_number = len([f for f in self.source_files if os.path.isfile(os.path.join(self.source_path, f))])
        self.target_image_number = len([f for f in self.target_files if os.path.isfile(os.path.join(self.target_path, f))]) 
        self.image_labels = getImageLabels(image_number=self.source_image_number,
                                           source_path=self.source_path,
                                           target_path=self.target_path)
        
    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self, index):
        def getSourceImagePath(index):
            return self.image_labels[0][index]
        def getTargetImagePath(index):
            return self.image_labels[1][index]
        source_image_path = getSourceImagePath(index)
        target_image_path = getTargetImagePath(index)
        source_image = io.imread(source_image_path).astype(np.float32) / 255.0
        target_image = io.imread(target_image_path).astype(np.float32) / 255.0
        source_image_tensor = torch.from_numpy(source_image)
        target_image_tensor = torch.from_numpy(target_image)
        sample = {"source_image": source_image_tensor, "target_image": target_image_tensor}
        return sample
    



def main():
    root = "./dataset"
    source_folder = "LF"
    target_folder = "GT"
    dataloader = HyperResolutionDataLoader(image_root_path=root,
                                           image_source_folder=source_folder,
                                           image_target_folder=target_folder)
    # print(dataloader.image_labels)
    # image_dic = dataloader.__getitem__(1)
    # print(type(image_dic["source_image"]))
    # print(image_dic["source_image"].shape)

if __name__ == "__main__":
    main()