from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import cv2
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

import src.data_loader as loader

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


data_img = loader.CommonDataset("image_test","train/lower_body/", "jpg")
print(len(data_img))
# print(data_img[0])
# fig= plt.figure()
#
# for i in range(len(data_img)):
#     sample = data_img[i]
#     ax = plt.subplot(1, 4, i+1)
#     plt.tight_layout()
#     ax.set_title('sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample)
#     plt.pause(0.0001)


######################################################################
# Iterating through the dataset
# -----------------------------
#
# Let's put this all together to create a dataset with composed
# transforms.
# To summarize, every time this dataset is sampled:
#
# -  An image is read from the file on the fly
# -  Transforms are applied on the read image
# -  Since one of the transforms is random, data is augmentated on
#    sampling
#
# We can iterate over the created dataset with a ``for i in range``
# loop as before.




transformed_dataset = loader.CommonDataset(root_dir="image_test", sub_dir="train/lower_body/",
                                           file_type="jpg",
                                           transform=transforms.Compose([loader.Rescale(256),
                                                                          loader.RandomCrop(224),
                                                                          loader.ToTensor()]))


print(len(transformed_dataset))
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(sample.type())
    print(i, sample.size())

    if i == 3:
        break
######################################################################
# However, we are losing a lot of features by using a simple ``for`` loop to
# iterate over the data. In particular, we are missing out on:
#
# -  Batching the data
# -  Shuffling the data
# -  Load the data in parallel using ``multiprocessing`` workers.
#
# ``torch.utils.data.DataLoader`` is an iterator which provides all these
# features. Parameters used below should be clear. One parameter of
# interest is ``collate_fn``. You can specify how exactly the samples need
# to be batched using ``collate_fn``. However, default collate should work
# fine for most use cases.



data_loader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

"""""""""
DataLoader
"""""""""

fig= plt.figure()


for i_batch, sample_batched in enumerate(data_loader):
    sample = sample_batched
    print(sample)
    # Make a grid from batch
    out = torchvision.utils.make_grid(sample)


    plt.ioff()
    imshow(out)
    plt.pause(0.0001)







