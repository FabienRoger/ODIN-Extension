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
from math import ceil
from typing import Any, Literal, Union
import torch
from torch.autograd import Variable
import numpy as np
import numpy as np
import time
from tqdm import tqdm
from constants import NORM_SCALE, file_name
from itertools import islice
from torch.utils.data import DataLoader


class Algorithm(ABC):
    @abstractmethod
    def apply(self, images, net) -> list[tuple[float, str]]:
        ...

    @property  # type: ignore
    @abstractmethod
    def name(self) -> str:
        ...

    def blindInit(self, net, testLoaderIn, cuda_device: int = 0):
        pass


class BaseAlgorithm(Algorithm):
    def __init__(self, temperature: float = 1.0, reverse: bool = False, name: str = "Base"):
        self.temperature = temperature
        self._name = name
        self.reverse = reverse

    @torch.no_grad()
    def apply(self, images, net):
        outputs = net(images)
        if self.reverse:
            outputs = -outputs
        nnOutputs = np.max(logits_to_logprobs(outputs, self.temperature), axis=-1)
        return [(nnOutputs[i], "0") for i in range(len(nnOutputs))]

    @property
    def name(self) -> str:
        return self._name


class OdinAlgorithm(Algorithm):
    def __init__(self, temperature: float, noiseMagnitude: float, iters: int = 1, name: str = "Odin"):
        self.temperature = temperature
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

            # Using temperatureature scaling
            outputs = outputs / self.temperature
            nnOutputs = logits_to_logprobs(outputs)
            maxIndexTemp = np.argmax(nnOutputs, axis=-1)

            labels = Variable(torch.tensor(maxIndexTemp, device=outputs.device, dtype=torch.long))

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
            nnOutputs = np.max(logits_to_logprobs(outputs, self.temperature), axis=-1)
            return [(nnOutputs[i], "0") for i in range(len(nnOutputs))]

    @property
    def name(self) -> str:
        return self._name


class TempBlindInit(Algorithm):
    SAMPLES = 1024

    def __init__(self, parent: Union[OdinAlgorithm, BaseAlgorithm], name=None):
        self.parent = parent
        self._name = name or f"TempBlind {parent.name}"

    def apply(self, images, net):
        return self.parent.apply(images, net)

    @property
    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def blindInit(self, net, testLoaderIn: DataLoader, cuda_device: int = 0):
        dataset = testLoaderIn.dataset
        ds_len = len(dataset)  # type: ignore
        batch_size: int = testLoaderIn.batch_size  # type: ignore
        samples = np.random.choice(ds_len, self.SAMPLES, replace=False)
        logits_list = []
        batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
        for batch in batches:
            images = torch.stack([dataset[i][0] for i in batch], dim=0).cuda(cuda_device)
            logits_list.append(net(images))
        logits = torch.cat(logits_list, dim=0)

        temp = 0.5

        def is_temp_to_low():
            logprobs = torch.log_softmax(logits / temp, dim=-1)
            max_logprobs, _ = torch.max(logprobs, dim=-1)
            top_1percent = torch.quantile(max_logprobs, 0.99).item()
            return top_1percent > np.log(0.5)

        while is_temp_to_low():
            temp *= 2

        self.parent.temperature = temp


@torch.no_grad()
def logits_to_logprobs(logits: torch.Tensor, temperatureature: float = 1) -> np.ndarray:
    """Takes torch logits on gpu, returns numpy probs on cpu"""
    return torch.log_softmax(logits / temperatureature, dim=-1).detach().cpu().numpy()


def testData(
    net,
    CUDA_DEVICE,
    testLoaderIn: DataLoader,
    testLoaderOut: DataLoader,
    nnName,
    dataName,
    algorithms: list[Algorithm],
    maxImages=128,
    skipFirstImages=1024,
):

    for alg in algorithms:
        alg.blindInit(net, testLoaderIn, CUDA_DEVICE)

    for testLoader, part in zip([testLoaderIn, testLoaderOut], ["In", "Out"]):
        print(f"Processing {part}-distribution images")

        batch_size = testLoader.batch_size
        assert batch_size is not None
        maxBatches = ceil(maxImages / batch_size)
        skipFirstBatchs = ceil(skipFirstImages / batch_size)

        files = {alg.name: open(file_name(nnName, dataName, alg.name, part), "w") for alg in algorithms}

        N = min(len(testLoader) - skipFirstBatchs, maxBatches)
        iterator = enumerate(islice(testLoader, skipFirstBatchs, skipFirstBatchs + N))
        print(f"Processing {N} batches of {batch_size} images each for a total of {N * batch_size} images")

        for j, data in tqdm(iterator, total=N):
            images, _ = data  # (batch_size, 3, 32, 32)
            images = images.cuda(CUDA_DEVICE)

            for alg in algorithms:
                scores = alg.apply(images, net)
                for score, metadata in scores:
                    files[alg.name].write(f"{score},{metadata}\n")  # 0 to force 2D data
        for f in files.values():
            f.close()
