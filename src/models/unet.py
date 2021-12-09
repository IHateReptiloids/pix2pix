import torch
import torch.nn as nn
import torchvision

from src.utils import pairwise


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        padding,
        stride,
        relu_slope,
        dropout,
    ):
        super().__init__()
        self.first_layer = DownBlock(
            in_channels,
            hidden_channels[0],
            kernel_size,
            padding,
            stride=1,
            relu_slope=relu_slope,
            batch_norm=False
        )

        self.encoder = nn.ModuleList()
        batch_norm = False
        hidden_channels = list(hidden_channels)
        for in_c, out_c in pairwise(hidden_channels + [hidden_channels[-1]]):
            self.encoder.append(
                DownBlock(in_c, out_c, kernel_size, padding, stride,
                          relu_slope, batch_norm)
            )
            batch_norm = True

        self.decoder = nn.ModuleList()
        for (in_c_div2, out_c), dropout_ in zip(
            pairwise([hidden_channels[-1] // 2] + hidden_channels[::-1]),
            dropout
        ):
            self.decoder.append(
                UpBlock(in_c_div2 * 2, out_c, kernel_size, stride, dropout_)
            )

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_channels[0], out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        '''
        x is of shape (bs, c, h, w)
        '''
        x = self.first_layer(x)
        outputs = []
        for enc in self.encoder:
            x = enc(x)
            outputs.append(x)

        x = torch.empty(x.shape[0], 0, x.shape[2], x.shape[3]) \
            .to(outputs[-1].device)
        for dec, copied in zip(self.decoder, outputs[::-1]):
            x = torch.cat((copied, x), dim=1)
            expected_shape = tuple(map(lambda elem: elem * 2, x.shape[-2:]))
            x = dec(x)
            x = torchvision.transforms.functional \
                .center_crop(x, expected_shape)
        return self.final_layer(x)

    @classmethod
    def from_config(cls, config):
        return cls(
            config.in_channels,
            config.out_channels,
            config.hidden_channels,
            config.kernel_size,
            config.padding,
            config.stride,
            config.relu_slope,
            config.dropout
        )


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
