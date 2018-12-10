from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
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
######################################################################
# Afterword: torchvision
# ----------------------
#
# In this tutorial, we have seen how to write and use datasets, transforms
# and dataloader. ``torchvision`` package provides some common datasets and
# transforms. You might not even have to write custom classes. One of the
# more generic datasets available in torchvision is ``ImageFolder``.
# It assumes that images are organized in the following way: ::
#
#     root/ants/xxx.png
#     root/ants/xxy.jpeg
#     root/ants/xxz.png
#     .
#     .
#     .
#     root/bees/123.jpg
#     root/bees/nsdf3.png
#     root/bees/asd932_.png
#
# where 'ants', 'bees' etc. are class labels. Similarly generic transforms
# which operate on ``PIL.Image`` like  ``RandomHorizontalFlip``, ``Scale``,
# are also avaiable. You can use these to write a dataloader like this: ::
#
#   import torch
#   from torchvision import transforms, datasets
#
#   data_transform = transforms.Compose([
#           transforms.RandomSizedCrop(224),
#           transforms.RandomHorizontalFlip(),
#           transforms.ToTensor(),
#           transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#       ])
#   hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                              transform=data_transform)
#   dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                                batch_size=4, shuffle=True,
#                                                num_workers=4)
#
# For an example with training code, please see
# :doc:`transfer_learning_tutorial`.

data_transform = transforms.Compose([

          transforms.RandomSizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
      ])

data_dir = "image_test"

torchvision_transformed_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"),transform=data_transform)

data_loader = torch.utils.data.DataLoader(torchvision_transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

fig= plt.figure()


print(torchvision_transformed_dataset.classes) # ['lower_body', 'upper_body']
for i_batch, sample_batched in enumerate(data_loader):
    if i_batch ==2:
        break
    data, label = sample_batched

    # sample = sample.type(torch.FloatTensor)
    # Make a grid from batch
    out = torchvision.utils.make_grid(data, normalize=True)

    plt.ioff()
    imshow(out, label)
    plt.pause(0.0001)
