import random
from typing import List

import torch 
import torch.nn as nn

from ..builder import FUSION_BEV_ENCODER

@FUSION_BEV_ENCODER.register_module()
class BEVAddFuser(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.transforms = nn.ModuleList()
        for k in range(len(in_channels)):
            self.transforms.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[k], out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        features = []
        for transform, input in zip(self.transforms, inputs):
            features.append(transform(input))

        weights = [1] * len(inputs)
        if self.training and random.random() < self.dropout:
            index = random.randint(0, len(inputs) - 1)
            weights[index] = 0

        return sum(w * f for w, f in zip(weights, features)) / sum(weights)