from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

import src.data_loader as loader


data_img = loader.CommonDataset("image_test","train/lower_body/", "jpg")
print(len(data_img))

fig= plt.figure()

for i in range(len(data_img)):
    sample = data_img[i]
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample)
    plt.pause(0.0001)
