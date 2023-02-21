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
from abc import abstractmethod, ABC
from typing import Any
import torch
from torch.autograd import Variable
import numpy as np
import numpy as np
import time
from tqdm import tqdm
from constants import NORM_SCALE, file_name
from itertools import islice


class Algorithm(ABC):
    @abstractmethod
    def apply(self, images, net) -> list[tuple[float, str]]:
        ...

    @property  # type: ignore
    @abstractmethod
    def name(self) -> str:
        ...


class BaseAlgorithm(Algorithm):
    @torch.no_grad()
    def apply(self, images, net):
        outputs = net(images)
        nnOutputs = np.max(logits_to_probs(outputs))  # TODO: make batchable
        return [(nnOutputs, "0")]

    @property
    def name(self) -> str:
        return "Base"


class OdinAlgorithm(Algorithm):
    def __init__(self, temper, noiseMagnitude, iters=1, name="Odin"):
        self.temper = temper
        self.noiseMagnitude = noiseMagnitude
        self.iters = iters
        self.criteria = torch.nn.CrossEntropyLoss()
        self._name = name

    def apply(self, images, net):
        inputs = Variable(images, requires_grad=True)
        opt = torch.optim.SGD([inputs], lr=1e-3)  # just here to zero
        for _ in range(self.iters):
            opt.zero_grad()
            outputs = net(inputs)

            # Using temperature scaling
            outputs = outputs / self.temper
            nnOutputs = logits_to_probs(outputs)
            maxIndexTemp = np.argmax(nnOutputs)

            labels = Variable(torch.tensor([maxIndexTemp], device=outputs.device, dtype=torch.long))

            loss = self.criteria(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            norm_scale = torch.tensor(NORM_SCALE, device=gradient.device).view(1, 3, 1, 1)
            gradient /= norm_scale
            # Adding small perturbations to images
            inputs.data = torch.add(inputs.data, gradient, alpha=-self.noiseMagnitude)

        with torch.no_grad():
            outputs = net(inputs)
            # Calculating the confidence after adding perturbations
            nnOutputs = logits_to_probs(outputs, self.temper)
            return [(np.max(nnOutputs), "0")]  # TODO: make batchable

    @property
    def name(self) -> str:
        return self._name


def logits_to_probs(logits: torch.Tensor, temperature=1) -> np.ndarray:
    """Takes torch logits on gpu, returns numpy probs on cpu

    TODO: make it batchable"""
    logits = logits / temperature
    nnOutputs = logits.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
    return nnOutputs


def testData(
    net,
    CUDA_DEVICE,
    testLoaderIn,
    testLoaderOut,
    nnName,
    dataName,
    algorithms: list[Algorithm],
    maxImages=100,
    skipFirstImages=1000,
):

    for testLoader, part in zip([testLoaderIn, testLoaderOut], ["In", "Out"]):
        print(f"Processing {part}-distribution images")

        files = {alg.name: open(file_name(nnName, dataName, alg.name, part), "w") for alg in algorithms}

        N = min(len(testLoader) - skipFirstImages, maxImages)
        iterator = enumerate(islice(testLoader, skipFirstImages, skipFirstImages + N))

        for j, data in tqdm(iterator, total=N):
            images, _ = data  # (batch_size, 3, 32, 32), where batch_size = 1
            images = images.cuda(CUDA_DEVICE)

            for alg in algorithms:
                scores = alg.apply(images, net)
                for score, metadata in scores:
                    files[alg.name].write(f"{score},{metadata}\n")  # 0 to force 2D data
        for f in files.values():
            f.close()
