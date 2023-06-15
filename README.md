# Image Segmentation for Counting Lamellipodia
This repository uses machine learning for the computer vision problem of image segmentation to count the number of lamellipodia on canine epithelial cells infected with Canid herpesvirus 1. 

## Project Description
The plasma membrane of mammalian cells forms the boundary between the cell and its environment and is critical for several biological functions. Neither the composition nor the structure of this membrane are static. Nutrient changes in the cell's environment as well as exposure to pathogens can induce changes in this cell membrane. In particular, the interaction of several viruses with the outer surface of the plasma membrane leads to the formation of membrane extensions. These extensions are typically induced by viruses to promote virus entry into the host cell. Scanning electron microscopy makes it possible to visualize the cell surface and its ultrastructure. Thus we can distinguish various types of extensions of the plasma membrane. However, quantitative analysis of such virus-induced changes can be very challenging. A cell can have thousands of extensions. Moreover, to make a complete study, it is necessary to analyze several cells within the framework of several independent experiments. 

### Objective
The objective of this project is to apply machine learning tools to analyze and classify images of canine epithelial cells that have or have not been infected with Canid herpesvirus 1. This virus induces the formation of lamellipodia-like membrane extensions. A machine learning strategy will be established which would allow for the quantitative analysis of the density of these extensions. Such a tool would allow the analysis of the impact of the virus under various conditions on the ultrastructure of the plasma membrane of the target cell. This information will contribute to a better understanding of the interaction of this virus with the host. This strategy could also be used to study other veterinary and human viruses.

## Methodology

### SEM image example
**Manual Count**: 40

![image](https://github.com/lhui2001/machine-learning-sem/assets/96440609/d5ec4672-2610-4a69-b677-c90ede616ff2)

### Computer Vision
A simple solution is to classify pixels in SEM images based on their intensity. 

#### a) Edge detection
**Count**: 39

![image](https://github.com/lhui2001/machine-learning-sem/assets/96440609/115cb492-c004-4243-8493-35f7c42894ab)

#### b) Watershed segmentation
**Count**: 43

![image](https://github.com/lhui2001/machine-learning-sem/assets/96440609/230722a1-f0db-46d1-8300-d65db4419054)

### U-NET
[U-NET](https://arxiv.org/abs/1505.04597v1) is a convolutional neural network used for biomedical image segmentation.  

#### step 1) Semantic segmentation
**IoU score**: 0.8374839004357684

![image](https://github.com/lhui2001/machine-learning-sem/assets/96440609/3f7ac103-d685-46f0-8b00-1dbd274f78f5)

#### step 2 (a) Instance segmentation
`Number lamellipodia = Sum of segmentation area  / Average area of lamellipodia`

**Count**: 38

![image](https://github.com/lhui2001/machine-learning-sem/assets/96440609/570cadc9-001c-4345-a6f8-897877d0c4f7)


#### step 2 (b) Watershed segmentation
**Count**: 35

![image](https://github.com/lhui2001/machine-learning-sem/assets/96440609/9a9ae812-96de-41ed-bc0c-aec91fb63c73)
