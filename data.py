import torch
import os
import numpy as np
#import imageio
#import tifffile
import cv2
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision.transforms import ToTensor

class SeepDataset():
    """
    Dataloader for our dataset
    """

    def __init__(self, input_dir, mask_dir, many_classes):
        self.many = many_classes
        inputs = os.listdir(input_dir)

        images = []
        masks = []
#[0.0000, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196, 0.0235, 0.0275]
        for im in inputs:
            img = cv2.imread(input_dir+'/'+im, -1)
            image = ToTensor()(cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX).astype(np.float32))*255
            mask = ToTensor()(Image.open(mask_dir+'/'+im))
            mask2 = mask.clone()
            if self.many:
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals = 4) == 0.0039] = 1
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals = 3) == 0.008] = 2
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals = 3) == 0.01] = 3
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals = 4) == 0.0157] = 4
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals = 4) == 0.0196] = 5
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals = 4) == 0.0235] = 6
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals = 4) == 0.0275] = 7
            else:
                 mask2[0][np.around(mask[0][0].detach().numpy(), decimals=4) > 0] = 1
            images.append(image.float())
            masks.append(mask2[0].long())
        self.input = torch.stack(images)
        self.output = torch.stack(masks)
        
    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        inputs = self.input[idx, :]
        outputs = self.output[idx]
        return inputs, outputs



    def get_images(self):
    	return self.input

    def get_masks(self):
    	return self.output
