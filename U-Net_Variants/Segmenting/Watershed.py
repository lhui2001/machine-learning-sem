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

# load images and predict mask
im_tb = Image.open("C:/Users/lucyh/Desktop/Images/Cells/Images/Mock 2 T0-0008-5k_#path788.tiff").convert('RGB')
s_tb = data_transform(im_tb)
pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
pred_tb = F.sigmoid(pred_tb)
pred_tb = pred_tb.detach().numpy()
pred_tb = pred_tb[0, 0, :, :]
# plt.imshow(pred_tb, cmap='gray')

# put threshold on mask to make it binary
binarr = np.where(pred_tb>0.5, 1, 0)
plt.imshow(binarr, cmap='gray')

# save output
plt.imsave('C:/.../NestedUNet.jpg', binarr, cmap='gray')
