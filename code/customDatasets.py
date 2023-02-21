import torch
from torch.utils.data import Dataset
from constants import NORM_BIAS, NORM_SCALE


class UniformNoiseDataset(Dataset):
    def __init__(self, size=10000, channels=3, height=32, width=32, norm_bias=NORM_BIAS, norm_scale=NORM_SCALE):
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width
        self.norm_bias = torch.tensor(norm_bias).view(3, 1, 1)
        self.norm_scale = torch.tensor(norm_scale).view(3, 1, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        images = torch.rand(3, 32, 32)
        images = (images - self.norm_bias) / self.norm_scale
        dummy_label = torch.tensor(0)
        return images, dummy_label


class GaussianNoiseDataset(Dataset):
    def __init__(self, size=10000, channels=3, height=32, width=32, norm_bias=NORM_BIAS, norm_scale=NORM_SCALE):
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width
        self.norm_bias = torch.tensor(norm_bias).view(3, 1, 1)
        self.norm_scale = torch.tensor(norm_scale).view(3, 1, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        images = torch.randn(3, 32, 32) + 0.5
        images = torch.clamp(images, 0, 1)
        images = (images - self.norm_bias) / self.norm_scale
        dummy_label = torch.tensor(0)
        return images, dummy_label
