# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import argparse
from email.policy import default
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# import matplotlib.pyplot as plt
import numpy as np
import time

# import lmdb
from scipy import misc
import cal as c


parser = argparse.ArgumentParser(description="Pytorch Detecting Out-of-distribution examples in neural networks")

parser.add_argument("--nn", default="densenet10", type=str, help="neural network name and training set")
parser.add_argument("--out_dataset", default="Imagenet", type=str, help="out-of-distribution dataset")
parser.add_argument("--magnitude", default=0.0014, type=float, help="perturbation magnitude")
parser.add_argument("--temperature", default=1000, type=int, help="temperature scaling")
parser.add_argument("--max_images", default=100, type=int, help="max number of images to test")
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--only_metric", action="store_true", help="only calculate metric")
parser.set_defaults(argument=True)


# Setting the name of neural networks

# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Wide-ResNet trained on CIFAR-10:    wideresnet10
# Wide-ResNet trained on CIFAR-100:   wideresnet100
# nnName = "densenet10"

# Setting the name of the out-of-distribution dataset

# Tiny-ImageNet (crop):     Imagenet
# Tiny-ImageNet (resize):   Imagenet_resize
# LSUN (crop):              LSUN
# LSUN (resize):            LSUN_resize
# iSUN:                     iSUN
# Gaussian noise:           Gaussian
# Uniform  noise:           Uniform
# FSGM:                     FSGM attack on the train data
# dataName = "Imagenet"


# Setting the perturbation magnitude
# epsilon = 0.0014

# Setting the temperature
# temperature = 1000
def main():
    global args
    args = parser.parse_args()
    c.test(
        args.nn,
        args.out_dataset,
        args.magnitude,
        args.temperature,
        args.max_images,
        args.batch_size,
        args.only_metric,
    )


if __name__ == "__main__":
    main()
