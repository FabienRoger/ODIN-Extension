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
import os
from datetime import datetime

# import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import misc
from sklearn.metrics import roc_auc_score, roc_curve
from constants import file_name


def fpr05(labels, data):
    # calculate the true positive rate when fpr is <=5%
    fpr, tpr, _ = roc_curve(labels, data)
    fpr05_index = np.where(fpr <= 0.05)[0][-1]
    return tpr[fpr05_index]


def tpr95(labels, data):
    # calculate the false positive rate when tpr is >= 95%

    fpr, tpr, _ = roc_curve(labels, data)
    tpr95_index = np.where(tpr >= 0.95)[0][0]
    return fpr[tpr95_index]


def auroc(labels, data):
    return roc_auc_score(labels, data)


def metric(nn, dsName, algorithms):
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

    methods = {
        "FPR at TPR 95%:": (tpr95),
        "AUROC:": auroc,
        "TPR at FPR 5%*:": (fpr05),
    }

    algnames = [alg.name for alg in algorithms]

    methods_res = {method_name: [] for method_name in methods.keys()}

    for alg_name in algnames:
        datas = [np.loadtxt(file_name(nn, dsName, alg_name, part), delimiter=",")[:, 0] for part in ["In", "Out"]]
        assert datas[0].shape == datas[1].shape
        data = np.concatenate(datas)
        labels = np.concatenate([np.ones(len(data) // 2), np.zeros(len(data) // 2)])
        for method_name, method in methods.items():
            methods_res[method_name].append(method(labels, data))

    text = [
        "{:31}{:>22}".format("Neural network architecture:", nnStructure),
        "{:31}{:>22}".format("Out-of-distribution dataset:", dsName),
        "{:31}{:>22}".format("Datapoints in distribution:", len(datas[0])),  # type: ignore
        "{:31}{:>22}".format("Datapoints out distribution:", len(datas[1])),  # type: ignore
        "",
        f"{'Method':15}|" + "|".join([f"{alg_name:>14}" for alg_name in algnames]),
    ]
    for method_name, method_res in methods_res.items():
        text.append(f"{method_name:15}|" + "|".join([f"{res * 100:>13.1f}%" for res in method_res]))

    # Print
    for line in text:
        print(line)
    # Save
    os.makedirs("results", exist_ok=True)
    with open(f"results/{nn}_{dsName}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.out", "w") as f:
        f.write("\n".join(text))
