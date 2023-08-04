# convert tiff to png 
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

path = "C:/Users/lucyh/Desktop/Single Class/Images/2023-05-25/Images/"
for file in os.listdir(path):
    if file.endswith(".tiff"):
        img = Image.open(path+str(file))
        img = np.array(img)  
        #binarr = np.where(img>0, 255, 0)  #change img to binary
        #binimg = Image.fromarray(binarr)
        cv2.imwrite(path+file[0:-4]+"png", img)
        os.remove(path+file)
