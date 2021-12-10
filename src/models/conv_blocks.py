import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        relu_slope,
        batch_norm: bool,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            (nn.BatchNorm2d(out_channels, track_running_stats=False)
                if batch_norm else nn.Identity()),
            nn.LeakyReLU(relu_slope, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dropout,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
