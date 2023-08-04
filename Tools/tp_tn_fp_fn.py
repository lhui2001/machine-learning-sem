############################ calculate for individual images ############################
import numpy as np 
from matplotlib import image
from matplotlib import pyplot

true_labels = image.imread("C:/Users/lucyh/Desktop/Single Class/Images/2023-05-25/Masks/Mock 2 T0-0002-5k_#path610.png")
true_labels = true_labels[:,:,0]
pred_labels = image.imread("C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/AttU_Net_mask/Mock 2 T0-0002-5k_#path610.png")

TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))  
print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))


############################ calculate for images in a directory ############################
import numpy as np 
from matplotlib import image
from matplotlib import pyplot
import pandas as pd
import os
import cv2 

def main():
    
    outPath = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/[0] Excel/"
    true = "C:/Users/lucyh/Desktop/Single Class/Images/2023-05-25/Masks/"
    pred = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/U_Net_mask/"
    
    tp = []
    tn = []
    fp = []
    fn = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    file = []

    # iterate through the images of the folder
    for line1 in os.listdir(true):
            
        # read image
        path1 = os.path.join(true, line1)
        path2 = os.path.join(pred, line1)
        true_labels = image.imread(path1)
        true_labels = true_labels[:,:,0]
        true_labels = cv2.resize(true_labels, (128, 128))
        pred_labels = image.imread(path2)
            
        # TP TN FP FN
        TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))  
        TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))  
        FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))  
        FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))  
            
        # accuracy precision recall f1
        Accuracy = (TP+TN)/(TP+TN+FP+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*TP/(2*TP+FP+FN)
            
        # dataframe
        tp.append(TP)
        tn.append(TN)
        fp.append(FP)
        fn.append(FN)
        accuracy.append(Accuracy)
        precision.append(Precision)
        recall.append(Recall)
        f1.append(F1)
        file.append(line1)
            
    #export
    df = pd.DataFrame(zip(file, tp, tn, fp, fn, accuracy, precision, recall, f1), columns = ['File', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1'])
    df.to_excel(outPath + "U_Net_metrics.xlsx", index=False) 
    
if __name__ == '__main__':
    main()


############################ calculate mean and SD values ############################

import pandas as pd

def main():
    
    outPath = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/[0] Excel/metric"
    path = "C:/Users/lucyh/Desktop/Single Class/All unet models/Predictions/[0] Excel/metric"
    
    folder = []
    aa = []
    ap = []
    ar = []
    af = []
    sa = []
    sp = []
    sr = []
    sf = []
    
    # iterate through excel files of the folder
    for file in os.listdir(path):

        # create the full input path and read the image
        input_path = os.path.join(path, file)
        df = pd.read_excel(input_path)
        
        # calculate mean
        Aa = df["Accuracy"].mean()
        Ap = df["Precision"].mean()
        Ar = df["Recall"].mean()
        Af = df["F1"].mean()
        Sa = df["Accuracy"].sem()
        Sp = df["Precision"].sem()
        Sr = df["Recall"].sem()
        Sf = df["F1"].sem()
        
        # save values
        folder.append(file)
        aa.append(Aa)
        ap.append(Ap)
        ar.append(Ar)
        af.append(Af)
        sa.append(Sa)
        sp.append(Sp)
        sr.append(Sr)
        sf.append(Sf)
    
    # export values
    df = pd.DataFrame(zip(folder, aa, sa, ap, sp, ar, sr, af, sf), columns = ['File', 'Accuracy Mean', 'Accuracy SD', 'Precision Mean', 'Precision SD', 'Recall Mean', 'Recall SD', 'F1 Mean', 'F1 SD'])
    df.to_excel("Average_metrics.xlsx", index=False)
                    
if __name__ == '__main__':
    main()
