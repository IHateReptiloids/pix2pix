from pathlib import Path

from argparse_dataclass import ArgumentParser
import torch
import torchvision
from tqdm import tqdm
import wandb

from src.configs import DefaultConfig
from src.datasets import FacadesDataset, Edges2ShoesDataset, get_dataloaders
from src.models import Pix2Pix, UNet
from src.utils import seed_all

DATASETS = {
    'facades': FacadesDataset,
    'edges2shoes': Edges2ShoesDataset
}

MODES = {
    'gan': Pix2Pix,
    'l1': UNet
}

STATE_DICT_KEY = {
    'gan': 'gan',
    'l1': 'model'
}

RESULT_DIR = Path('examples')


config = ArgumentParser(DefaultConfig).parse_args()
seed_all(config.random_seed)

ds_cls = DATASETS[config.dataset]
model_cls = MODES[config.mode]

val_ds = ds_cls('val')
assert config.val_batch_size == 1
_, val_loader = get_dataloaders(None, val_ds, config)

model = model_cls.from_config(config).to(config.device).eval()

wandb.init(config=config, group=f'{config.dataset}-{config.mode}')
assert config.wandb_file_name is not None and config.wandb_run_path is not None
f = wandb.restore(config.wandb_file_name, config.wandb_run_path)
state_dict = torch.load(f.name, map_location=config.device)
model.load_state_dict(state_dict[STATE_DICT_KEY[config.mode]])
model.eval()

table = wandb.Table(
    columns=['val/input_image', 'val/target_image', 'val/output_image']
)
gt_dir = RESULT_DIR / config.mode / config.dataset / 'ground_truth'
out_dir = RESULT_DIR / config.mode / config.dataset / 'output'
for dir_ in (gt_dir, out_dir):
    dir_.mkdir(parents=True, exist_ok=True)

for i, (x, y) in enumerate(tqdm(val_loader)):
    out = model(x)
    x = x.squeeze().detach().cpu()
    y = y.squeeze().detach().cpu()
    out = out.squeeze().detach().cpu()
    torchvision.utils.save_image(y, gt_dir / f'{i}.jpg')
    torchvision.utils.save_image(out, out_dir / f'{i}.jpg')
    table.add_data(
        wandb.Image(x.permute(1, 2, 0).numpy(), mode='RGB'),
        wandb.Image(y.permute(1, 2, 0).numpy(), mode='RGB'),
        wandb.Image(out.permute(1, 2, 0).numpy(), mode='RGB')
    )
wandb.log({'val/results': table})
