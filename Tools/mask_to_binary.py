from scipy import ndimage, misc
import numpy as np
import os
import cv2
import imageio.v2 as imageio

def main():
    outPath = "C:/Users/"
    path = "C:/Users/"

    # iterate through the mask images of the folder
    for image_path in os.listdir(path):

        # create the full input path and read the image
        input_path = os.path.join(path, image_path)
        image_to_mask = imageio.imread(input_path)

        # change masks to binary 
        binary = np.where(image_to_mask>0, 255, image_to_mask)

        # create full output path
        fullpath = os.path.join(outPath, image_path)
        imageio.imwrite(fullpath, binary)

if __name__ == '__main__':
    main()
