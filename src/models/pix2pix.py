import torch.nn as nn

from .patch_gan import PatchGAN
from .unet import UNet


class Pix2Pix(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        unet_hidden_channels,
        pgan_hidden_channels,
        kernel_size,
        padding,
        stride,
        relu_slope,
        dropout,
    ):
        super().__init__()
        self.G = UNet(
            in_channels,
            out_channels,
            unet_hidden_channels,
            kernel_size,
            padding,
            stride,
            relu_slope,
            dropout
        )
        self.D = PatchGAN(
            in_channels + out_channels,
            pgan_hidden_channels,
            kernel_size,
            padding,
            stride,
            relu_slope
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            config.in_channels,
            config.out_channels,
            config.unet_hidden_channels,
            config.pgan_hidden_channels,
            config.kernel_size,
            config.padding,
            config.stride,
            config.relu_slope,
            config.dropout
        )
