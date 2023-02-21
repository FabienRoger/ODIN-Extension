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
from scipy import misc
from sklearn.metrics import roc_auc_score, roc_curve


def tpr95(labels, data):
    # calculate the falsepositive error when tpr is 95%

    fpr, tpr, thresholds = roc_curve(labels, data)
    tpr95_index = np.where(tpr >= 0.95)[0][0]
    return fpr[tpr95_index]


def auroc(labels, data):
    return roc_auc_score(labels, data)


def metric(nn, data):
    if nn == "densenet10" or nn == "wideresnet10":
        indis = "CIFAR-10"
    elif nn == "densenet100" or nn == "wideresnet100":
        indis = "CIFAR-100"
    else:
        raise ValueError(f"Invalid neural network name {nn}")
    if nn == "densenet10" or nn == "densenet100":
        nnStructure = "DenseNet-BC-100"
    elif nn == "wideresnet10" or nn == "wideresnet100":
        nnStructure = "Wide-ResNet-28-10"
    else:
        raise ValueError(f"Invalid neural network name {nn}")

    if data == "Imagenet":
        dataName = "Tiny-ImageNet (crop)"
    elif data == "Imagenet_resize":
        dataName = "Tiny-ImageNet (resize)"
    elif data == "LSUN":
        dataName = "LSUN (crop)"
    elif data == "LSUN_resize":
        dataName = "LSUN (resize)"
    elif data == "iSUN":
        dataName = "iSUN"
    elif data == "Gaussian":
        dataName = "Gaussian noise"
    elif data == "Uniform":
        dataName = "Uniform Noise"
    else:
        raise ValueError(f"Invalid dataset name {data}")

    labels_and_data = []
    for algorithm in ["Base", "Our"]:
        datas = [
            np.loadtxt(f"./softmax_scores/confidence_{algorithm}_{part}.txt", delimiter=",")[:, 2]
            for part in ["In", "Out"]
        ]
        assert datas[0].shape == datas[1].shape
        data = np.concatenate(datas)
        labels = np.concatenate([np.ones(len(data) // 2), np.zeros(len(data) // 2)])
        labels_and_data.append((labels, data))

    base, our = labels_and_data
    fprBase, fprNew = tpr95(*base), tpr95(*our)
    aurocBase, aurocNew = auroc(*base), auroc(*our)
    print("{:31}{:>22}".format("Neural network architecture:", nnStructure))
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    print("")
    print("{:>34}{:>19}".format("Baseline", "Our Method"))
    print("{:20}{:13.1f}%{:>18.1f}% ".format("FPR at TPR 95%:", fprBase * 100, fprNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUROC:", aurocBase * 100, aurocNew * 100))
