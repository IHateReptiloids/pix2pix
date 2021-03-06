from argparse_dataclass import ArgumentParser
import torch

from src.configs import DefaultConfig
from src.datasets import FacadesDataset, Edges2ShoesDataset, get_dataloaders
from src.models import Pix2Pix, UNet
from src.trainers import GANTrainer, L1Trainer
from src.utils import seed_all

DATASETS = {
    'facades': FacadesDataset,
    'edges2shoes': Edges2ShoesDataset
}
MODES = {
    'gan': Pix2Pix,
    'l1': UNet
}


config = ArgumentParser(DefaultConfig).parse_args()
seed_all(config.random_seed)

ds_cls = DATASETS[config.dataset]
model_cls = MODES[config.mode]

train_ds = ds_cls('train')
val_ds = ds_cls('val')
train_loader, val_loader = get_dataloaders(train_ds, val_ds, config)

model = model_cls.from_config(config).to(config.device).train()
logger = None
if config.logging:
    from src.loggers import WandbLogger
    logger = WandbLogger(model, scheduler=None, config=config)
    from torchinfo import summary
    summary(model)

if config.mode == 'l1':
    opt = torch.optim.Adam(model.parameters(),
                           lr=config.lr, betas=config.betas)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    if logger is not None:
        logger.scheduler = scheduler
    trainer = L1Trainer(
        config,
        model,
        opt,
        scheduler,
        train_loader,
        val_loader,
        logger
    )
else:
    opt_d = torch.optim.Adam(model.D.parameters(),
                             lr=config.lr, betas=config.betas)
    opt_g = torch.optim.Adam(model.G.parameters(),
                             lr=config.lr, betas=config.betas)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda _: 1.0)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda _: 1.0)
    if logger is not None:
        logger.scheduler = scheduler_d
    trainer = GANTrainer(
        config,
        model,
        opt_g,
        opt_d,
        scheduler_g,
        scheduler_d,
        train_loader,
        val_loader,
        logger
    )

if config.wandb_file_name is not None and config.wandb_run_path is not None:
    import wandb
    f = wandb.restore(config.wandb_file_name, config.wandb_run_path)
    state_dict = torch.load(f.name, map_location=config.device)
    trainer.load_state_dict(state_dict)

trainer.train(config.num_epochs)
