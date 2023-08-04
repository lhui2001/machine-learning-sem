# create bounding boxes from masks

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

if __name__ == "__main__":
    """ Load the dataset """
    images = sorted(glob(os.path.join("data", "image", "*")))     # load images frome directory data containing directory image in current working directory
    masks = sorted(glob(os.path.join("data", "mask", "*")))       # load masks from directory data containing directory mask in current working directory

    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        name = os.path.splitext(os.path.basename(x))[0]
        print(name)
        
        """ Read image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Detecting bounding boxes """
        border = mask_to_border(y)

        """ Saving the image """
        cv2.imwrite(f"C:/.../{name}.png", border)
