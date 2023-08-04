# https://github.com/nikhilroxtomar/Semantic-Segmentation-Mask-to-Bounding-Box
# https://www.youtube.com/watch?v=RmLDL7AVXUc&t=684s&pp=ygUUbWFzayB0byBib3VuZGluZyBib3g%3D
# only works with png 

import pandas as pd
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours

def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

if __name__ == "__main__":
    """ Load the dataset """
    images = sorted(glob(os.path.join("data", "image", "*")))
    masks = sorted(glob(os.path.join("data", "mask", "*")))
    
    for x, y in tqdm(zip(images, masks), total=len(images)):
        
        """ Extract the name """
        name = os.path.splitext(os.path.basename(x))[0]
        
        """ Read image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        
        """ Detecting bounding boxes """
        bboxes = mask_to_bbox(y)
        
        """ Save to txt """
        df = pd.DataFrame(data=bboxes)
        np.savetxt(f"C:/.../{name}.txt", df.values, fmt='%d')
