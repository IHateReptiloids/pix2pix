import torch.nn as nn


class Pix2Pix(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
