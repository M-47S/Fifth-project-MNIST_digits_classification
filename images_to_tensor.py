import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class TransformImageToTensor():
    def __init__(self, path_to_dir_images: str = None):
        self.change_path(path_to_dir_images=path_to_dir_images)
    
    def change_path(self, path_to_dir_images: str = None):
        self.__folder = Path(path_to_dir_images)
    
    def convert_to_tensor(self) -> torch.Tensor:
        image_list = []
        
        for image_path in self.__folder.glob("*.png"):
            image = plt.imread(image_path, format="png")
            
            gray = 0.2989*image[:,:,0] + 0.5870*image[:,:,1] + 0.1140*image[:,:,2]
            
            image_tensor = torch.tensor(gray, dtype=torch.float32)
            
            image_list.append(image_tensor)
        
        return torch.stack(image_list, dim=0)
            
    def save_like_tensor_file(self, save_dir: str = None, fname: str = "Tensor_image"):
        if save_dir is None:
            try:
                save_dir = "./Tensors images"
                os.mkdir(save_dir)
            except FileExistsError:
                pass
       
        tensors = self.convert_to_tensor()
        
        torch.save(tensors, f"{save_dir}/{fname}.pt")

def extract_labels(path_to_dir_images: str):
    folder = Path(path_to_dir_images)
    label_list = []
    
    for image_path in folder.glob("*.png"):
        label = int(image_path.stem[-1])       
        label_list.append(torch.tensor(label))
     
    return torch.stack(label_list, dim=0)   
    
            
if __name__ == "__main__":
    tr = TransformImageToTensor("./Images")
    tr.save_like_tensor_file(save_dir="./Tensors images", fname="Batch1_X")
    
    y = extract_labels("./Images")
    torch.save(y, "./Tensors images/Batch1_Y.pt")



