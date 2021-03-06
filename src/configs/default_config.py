from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DefaultConfig:
    checkpointing_freq: int = 20
    dataset: str = 'facades'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging: bool = False
    mode: str = 'gan'
    random_seed: int = 3407
    wandb_file_name: str = None
    wandb_run_path: str = None
    # model params
    in_channels: int = 3
    out_channels: int = 3
    unet_hidden_channels: Tuple[int, ...] = (64, 128, 256, 512, 512, 512, 512)
    pgan_hidden_channels: Tuple[int, ...] = (64, 128, 256, 512)
    kernel_size: Tuple[int, int] = (4, 4)
    padding: Tuple[int, int, int, int] = (2, 1, 2, 1)
    stride: int = 2
    relu_slope: float = 0.2
    dropout: Tuple[float, ...] = (0.5, 0.5, 0.5, 0, 0, 0, 0)
    # Adam params
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    # Training params
    lambda_l1: float = 100.0
    epoch_num_iters: int = 2000
    num_epochs: int = 40
    train_batch_size: int = 1
    train_log_freq: int = 500
    train_num_workers: int = 0
    val_batch_size: int = 1
    val_log_freq: int = 33
    val_num_workers: int = 0
