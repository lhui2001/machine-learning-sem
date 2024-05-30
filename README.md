# Quantifying lamellipodia
Application of machine learning methods for the quantification of lamellipodia in scanning electron microscopy of Madin-Darby Canine Kidney Cells (MDCK)

## Background
The dynamic nature of plasma membranes support many biological functions that are vital to cell survival. Variations in environmental factors such as nutrient concentrations and pathogen exposure can alter membrane structures. Recent studies have revealed that viruses exploit membrane dynamics by inducing the formation of membrane protrusions, such as lamellipodia, in order to facilitate its entry, trafficking, and spread in host cells. Quantitative analysis of virus-induced changes in membrane structures represent an important avenue for studying viral pathogenesis. However, manual quantification of membrane extensions is a time-consuming task, prone to fatigue errors, and susceptible to observer bias. Hence, the objective of this study was to apply machine learning strategies to quantify the number of lamellipodia in scanning electron microscopy (SEM) images of Madin Darby Canine Kidney Cells (MDCK).

## Methods
We propose a Convolutional Neural Network-based approach widely used in medical image segmentation called U-Net. Data sets based on SEM images of MDCK cells under different conditions were produced. Binary masks of the images were obtained through manual annotation using CVAT. The supervised machine learning model was trained and tested using a dataset of 189 SEM images with corresponding binary masks. Objects of interest were localized and quantified with watershed segmentation. The performance of the model was compared against four other similar architectures: Attention U-Net (Attn U-Net), Recurrent Residual U-Net (R2 U-Net), Attention Recurrent Residual U-Net (Attn R2 U-Net), and Nested U-Net (U-Net++). The resulting approach was applied to compare the average density of lamellipodia on the plasma membrane of MDCK cells that were mock-infected or infected with Canid herpesvirus 1 (CHV-1).

## Results
Average count from all five machine learning methods were comparable to average manual counts. Counts from machine learning methods consistently had lower variability compared to manual counts. U-Net surpasses other U-Net variants in detection accuracy (F1 score = 92.4%), however the difference was not significant. Compared to manual quantification, U-Net-derived lamellipodia counts showed a mean difference of 5.6.

## Conclusion
We show the ability of U-Net to segment, detect, and quantify lamellipodia from SEM images of MDCK. Characterization of the impact of viral infections on the ultrastructure of target cells can not only improve understanding on virus-host interactions, but also lead to novel targets for antiviral strategies. The pre-trained model and annotated dataset are released to support further research.

## Using the code

## Resources
* https://github.com/HumanSignal/awesome-data-labeling
* https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets
* https://arxiv.org/pdf/1807.10165v1.pdf
* https://scikit-image.org/docs/dev/auto_examples/applications/plot_human_mitosis.html

