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
import warnings

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
from torch.utils.data import DataLoader
from constants import NORM_BIAS, NORM_SCALE
from customDatasets import UniformNoiseDataset, GaussianNoiseDataset
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

# CUDA_DEVICE = 0

start = time.time()
# loading data sets

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(NORM_BIAS, NORM_SCALE),
    ]
)


# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
# nnName = "densenet10"

# imName = "Imagenet"


criterion = nn.CrossEntropyLoss()


def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):

    net1 = torch.load("../models/{}.pth".format(nnName))
    optimizer1 = optim.SGD(net1.parameters(), lr=0, momentum=0)
    net1.cuda(CUDA_DEVICE)

    if nnName == "densenet10" or nnName == "wideresnet10":
        testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)
        testloaderIn = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    elif nnName == "densenet100" or nnName == "wideresnet100":
        testset = torchvision.datasets.CIFAR100(root="../data", train=False, download=True, transform=transform)
        testloaderIn = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    else:
        raise ValueError(f"nnName is not correct: {nnName}")

    if dataName == "Uniform":
        testsetout = UniformNoiseDataset()
    elif dataName == "Gaussian":
        testsetout = GaussianNoiseDataset()
    else:
        testsetout = torchvision.datasets.ImageFolder("../data/{}".format(dataName), transform=transform)

    testloaderOut = DataLoader(testsetout, batch_size=1, shuffle=False, num_workers=2)

    d.testData(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature)
    m.metric(nnName, dataName)
