

import torch
from torchvision import datasets

def __getitem__(self,index):
        # Original ImageFolder  
    original_tuple = super(ImagePathFolder, self).__getitem__(index)
        # Image Path
    path = self.imgs[index][0]
        # New tuple that includes path
    path_tuple = (original_tuple + (path,))
    return path_tuple



