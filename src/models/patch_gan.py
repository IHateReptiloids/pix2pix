import torch
import torch.nn as nn

from .conv_blocks import DownBlock
from src.utils import pairwise


class PatchGAN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        padding,
        stride,
        relu_slope
    ):
        super().__init__()

        batch_norm = False
        hidden_channels = list(hidden_channels)
        layers = []
        for in_c, out_c in pairwise([in_channels] + hidden_channels):
            layers.append(DownBlock(in_c, out_c, kernel_size, padding,
                                    stride, relu_slope, batch_norm))
            batch_norm = True
        layers.append(nn.Conv2d(
            in_channels=hidden_channels[-1], out_channels=1,
            kernel_size=kernel_size, padding=padding)
        )
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat((x, y), dim=1))
