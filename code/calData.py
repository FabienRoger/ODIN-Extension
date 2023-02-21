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
import numpy as np
import numpy as np
import time
from tqdm import tqdm
from constants import NORM_SCALE


def logits_to_probs(logits, temperature=1):
    """Takes torch logits on gpu, returns numpy probs on cpu"""
    logits = logits / temperature
    nnOutputs = logits.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
    return nnOutputs


def testData(
    net1, criterion, CUDA_DEVICE, testLoaderIn, testLoaderOut, nnName, dataName, noiseMagnitude1, temper, max_n=100
):

    for testLoader, part in zip([testLoaderIn, testLoaderOut], ["In", "Out"]):
        print(f"Processing {part}-distribution images")
        f = open(f"./softmax_scores/confidence_Base_{part}.txt", "w")
        g = open(f"./softmax_scores/confidence_Our_{part}.txt", "w")
        N = min(len(testLoader), max_n)

        for j, data in tqdm(enumerate(testLoader), total=N):
            # if j < 1000:
            #     continue
            images, _ = data  # (batch_size, 3, 32, 32)

            inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad=True)
            # print(inputs.shape)
            outputs = net1(inputs)

            # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
            nnOutputs = logits_to_probs(outputs)
            f.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

            # Using temperature scaling
            outputs = outputs / temper

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            maxIndexTemp = np.argmax(nnOutputs)
            labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
            loss = criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            norm_scale = torch.tensor(NORM_SCALE).cuda(CUDA_DEVICE).view(1, 3, 1, 1)
            gradient /= norm_scale
            # Adding small perturbations to images
            tempInputs = torch.add(inputs.data, gradient, alpha=-noiseMagnitude1)
            outputs = net1(Variable(tempInputs))
            # Calculating the confidence after adding perturbations
            nnOutputs = logits_to_probs(outputs, temper)
            g.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

            if j == N - 1:
                break
        f.close()
        g.close()
