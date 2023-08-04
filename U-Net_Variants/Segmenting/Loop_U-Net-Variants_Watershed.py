######################## load model weights ######################## 

import numpy as np
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt

from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net

# available models
model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]

# define function
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

# change model input (0 = U_Net, 1 = R2U_Net, 2 = AttU_Net, 3 = R2AttU_Net, 4 = NestedUNet)
model_test = model_unet(model_Inputs[4], 3, 1)  

# load model weights
model_test.load_state_dict(torch.load('C:/.../Unet_epoch_100_batchsize_32.pth'))

# resize images
data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
         #   torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

# gpu
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")


######################## iterate and predict segmentation ########################

import os
import torch.nn.functional as F
import imageio.v2 as imageio

def main():
    outPath = "C:/.../NestedUNet/"
    os.makedirs(outPath)
    path = "C:/.../Images/"

    # iterate through the images of the folder
    for image_path in os.listdir(path):

        # create the full input path and read the image
        input_path = os.path.join(path, image_path)
        im_tb = Image.open(input_path).convert('RGB')
        
        # segment
        s_tb = data_transform(im_tb)
        pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
        pred_tb = F.sigmoid(pred_tb)
        pred_tb = pred_tb.detach().numpy()
        pred_tb = pred_tb[0, 0, :, :]

        # threshold 
        binarr = np.where(pred_tb>0.5, 1, 0)

        # save to folder 
        fullpath = os.path.join(outPath, image_path)
        imageio.imwrite(fullpath, binarr)

if __name__ == '__main__':
    main()

  
######################## watershed ########################

# only works on png images

import sys
import cv2
from matplotlib import pyplot as plt
import pyclesperanto_prototype as cle
from skimage import io
import pandas as pd
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from scipy import ndimage as ndi

def main():
    
    outPath = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/"
    path = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/U_Net_mask/"
    
    data = []
    image = []

    # iterate through the images of the folder
    for image_path in os.listdir(path):

        # create the full input path and read the image
        input_path = os.path.join(path, image_path)
        image_to_segment = plt.imread(input_path)
        
        # process image
        #img_grey = image_to_segment[:,:,0]
        
        # thresholding
        thresholds = filters.threshold_multiotsu(image_to_segment, classes=2)
        regions = np.digitize(image_to_segment, bins=thresholds)
        cells = image_to_segment >= thresholds[0]

        # watershed
        distance = ndi.distance_transform_edt(cells)
        local_max_coords = feature.peak_local_max(distance, min_distance=5)
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)
        segmented_cells = segmentation.watershed(-distance, markers, mask=cells)
        
        # count
        count = segmented_cells.max()

        # save count measurements
        image.append(image_path)
        data.append(count)

    # export measurements
    df = pd.DataFrame(list(zip(image , data)), columns =['Image', 'Number of lamellipodia'])
    basename_without_ext = os.path.basename(os.path.dirname(path)) 
    df.to_excel(outPath + basename_without_ext + "_Count.xlsx", index=False)
    
if __name__ == '__main__':
    main()


######################## calculate average and SD ########################

import pandas as pd

def main():
    
    outPath = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/[0] Excel/"
    path = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/[0] Excel/"
    
    folder = []
    x = []
    s = []
    
    # iterate through excel files of the folder
    for file in os.listdir(path):

        # create the full input path and read the image
        input_path = os.path.join(path, file)
        df = pd.read_excel(input_path)
        
        # calculate mean
        mean = df["Number of lamellipodia"].mean()
        sd = df["Number of lamellipodia"].std()
        
        # save values
        folder.append(file)
        x.append(mean)
        s.append(sd)
    
    # export values
    df = pd.DataFrame(list(zip(folder, x, s)), columns = ['File', 'Average number of lamellipodia', 'Standard deviation'])
    df.to_excel(outPath + "Average.xlsx", index=False)
                    
if __name__ == '__main__':
    main()
