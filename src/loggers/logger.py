import torch
import wandb


class WandbLogger:
    def __init__(self, model, scheduler, config):
        self.run = wandb.init(config=config,
                              group=f'{config.dataset}-{config.mode}')
        self.run.watch(model, log='all', log_freq=config.train_log_freq)

        self.scheduler = scheduler
        self.train_log_freq = config.train_log_freq
        self.val_log_freq = config.val_log_freq

        self._train_log_age = 0
        self._val_log_age = 0

        self._val_table = None

    def commit(self):
        if self._val_table is not None:
            self.run.log({'val/table': self._val_table},
                         step=self.scheduler.last_epoch)
        self._val_table = None
        self._val_log_age = 0

    @property
    def checkpoint_dir(self):
        return self.run.dir

    def log_batch(self, x, y, out, loss, train: bool):
        assert len(x) == len(y) == len(out)
        index = torch.randint(0, len(x), (1,)).item()
        x = x[index].detach().cpu().permute(1, 2, 0).numpy()
        y = y[index].detach().cpu().permute(1, 2, 0).numpy()
        out = out[index].detach().cpu().permute(1, 2, 0).numpy()
        if train:
            self.log_train_object(x, y, out, loss)
        else:
            self.log_val_object(x, y, out)

    def log_train_object(self, x, y, out, loss):
        self._train_log_age += 1
        data = {'train/loss': loss,
                'train/lr': self.scheduler.get_last_lr()[0]}
        if self._train_log_age == self.train_log_freq:
            self._train_log_age = 0
            data.update({
                'train/input_image': wandb.Image(x, mode='RGB'),
                'train/target_image': wandb.Image(y, mode='RGB'),
                'train/output_image': wandb.Image(out, mode='RGB')
            })
        self.run.log(data, step=self.scheduler.last_epoch)

    def log_val_object(self, x, y, out):
        self._val_log_age += 1
        if self._val_log_age == self.val_log_freq:
            self._val_log_age = 0
            if self._val_table is None:
                self._val_table = wandb.Table(columns=[
                    'val/input_image', 'val/target_image', 'val/output_image'
                ])
            self._val_table.add_data(
                wandb.Image(x, mode='RGB'),
                wandb.Image(y, mode='RGB'),
                wandb.Image(out, mode='RGB')
            )
