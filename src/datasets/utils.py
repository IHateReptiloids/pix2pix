import torch
from torch.utils.data import DataLoader


class VariableLengthLoader:
    def __init__(self, loader, num_iters):
        self.loader = loader
        self.num_iters = num_iters
        self._cur_iters = 0

    def __iter__(self):
        while True:
            for elem in self.loader:
                yield elem
                self._cur_iters += 1
                if self._cur_iters == self.num_iters:
                    self._cur_iters = 0
                    return

    def __len__(self):
        return self.num_iters


def get_dataloaders(train_ds, val_ds, config):
    train_loader = None
    if train_ds is not None:
        train_loader = DataLoader(
            train_ds, config.train_batch_size, shuffle=True,
            num_workers=config.train_num_workers, collate_fn=id_collator
        )
        train_loader = VariableLengthLoader(train_loader,
                                            config.epoch_num_iters)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, config.val_batch_size, shuffle=True,
            num_workers=config.val_num_workers, collate_fn=id_collator
        )
    return train_loader, val_loader


def id_collator(objects):
    '''
    stacks objects along batch axis
    '''
    x, y = list(zip(*objects))
    return torch.stack(x, dim=0), torch.stack(y, dim=0)
