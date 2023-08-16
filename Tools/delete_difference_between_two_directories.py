images = 'C:/Users/lucyh/Desktop/Single Class/images/training/images/'
masks = 'C:/Users/lucyh/Desktop/Single Class/images/training/masks/'

im = []
m = []

for image_path in os.listdir(images):
    im.append(image_path)

for mask_path in os.listdir(masks):
    m.append(mask_path)

difference = list(set(im) - set(m))

import os
path = 'C:/Users/lucyh/Desktop/Single Class/images/training/images/'
os.chdir(path)
for f in difference:
    fname = f.rstrip()
    if os.path.isfile(fname):
        os.remove(fname)
