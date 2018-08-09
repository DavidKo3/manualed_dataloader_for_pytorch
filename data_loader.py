# -*- coding: utf-8 -*-
"""
Data Loading and Processing Tutorial
====================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

A lot of effort in solving any machine learning problem goes in to
preparing the data. PyTorch provides many tools to make data loading
easy and hopefully, to make your code more readable. In this tutorial,
we will see how to load and preprocess/augment data from a non trivial
dataset.

To run this tutorial, please make sure the following packages are
installed:

-  ``scikit-image``: For image io and transforms
-  ``pandas``: For easier csv parsing

"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CommonDataset(Dataset):
    def __init__(self, root_dir , sub_dir, file_type, transform=None):
        """
        :param root_dir:  Path to the image datasets
        :param transform: Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.sub_dir = sub_dir
        self.file_type = file_type
        self.transform = transform
        self.img_name = os.path.join(self.root_dir, self.sub_dir)
        self.image_list = sorted(glob.glob(self.img_name + '*.' + self.file_type))
    def __len__(self):
        # img_name = os.path.join(self.root_dir, self.sub_dir)

        return len(self.image_list)

    def __getitem__(self, idx):
        img_item = io.imread(self.image_list[idx])

        if self.transform:
            img_item = self.transform(img_item)

        return img_item

######################################################################
# Transforms
# ----------
#
# One issue we can see from the above is that the samples are not of the
# same size. Most neural networks expect the images of a fixed size.
# Therefore, we will need to write some prepocessing code.
# Let's create three transforms:
#
# -  ``Rescale``: to scale the image
# -  ``RandomCrop``: to crop from image randomly. This is data
#    augmentation.
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it's
# called. For this, we just need to implement ``__call__`` method and
# if required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# Observe below how these transforms had to be applied both on the image and
# landmarks.
#
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int) : Desired output size. If int, square crop is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randit(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top+new_h, left:left+new_w]

        return image


class ToTensor():
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image : H x W x C
        # torch image : C x H x W
        image = image.transpose((2, 0, 1))
        return image




