from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits


class GANTrainer:
    def __init__(
        self,
        config,
        gan,
        opt_g,
        opt_d,
        scheduler_g,
        scheduler_d,
        train_loader,
        val_loader,
        logger=None
    ):
        self.device = config.device
        self.gan = gan
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.lambda_l1 = config.lambda_l1

        self.best_loss = None
        self.best_state = None

        self._checkpointing_freq = config.checkpointing_freq
        self._checkpoint_dir = None
        if self.logger is not None:
            self._checkpoint_dir = Path(self.logger.checkpoint_dir)

    def load_state_dict(self, state):
        self.gan.load_state_dict(state['gan'])
        self.opt_g.load_state_dict(state['opt_g'])
        self.opt_d.load_state_dict(state['opt_d'])
        self.scheduler_g.load_state_dict(state['scheduler_g'])
        self.scheduler_d.load_state_dict(state['scheduler_d'])

    def process_batch(self, x, y, train: bool):
        fake = self.gan.G(x)
        if train:
            self.gan.unfreeze_discriminator()
        d_fake = self.gan.D(x, fake.detach())
        d_real = self.gan.D(x, y)
        d_loss = (bce_logits(d_fake, torch.zeros_like(d_fake)) +
                  bce_logits(d_real, torch.ones_like(d_real))) / 2.0
        if train:
            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            self.scheduler_d.step()
            self.gan.freeze_discriminator()

        d_fake = self.gan.D(x, fake)
        g_loss = bce_logits(d_fake, torch.ones_like(d_fake)) + \
            F.l1_loss(fake, y) * self.lambda_l1
        if train:
            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()
            self.scheduler_g.step()

        total_loss = d_loss + g_loss
        if self.logger is not None:
            self.logger.log_batch(x, y, fake, total_loss.item(), train=train)
        return total_loss

    def state_dict(self):
        state = OrderedDict()
        state['gan'] = self.gan.state_dict()
        state['opt_g'] = self.opt_g.state_dict()
        state['opt_d'] = self.opt_d.state_dict()
        state['scheduler_g'] = self.scheduler_g.state_dict()
        state['scheduler_d'] = self.scheduler_d.state_dict()
        return state

    def train(self, num_epochs):
        for i in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            print(f'Epoch {i} train loss: {train_loss}')
            val_loss = self.validate()
            print(f'Epoch {i} validation loss: {val_loss}')
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state = deepcopy(self.state_dict())
                if self._checkpoint_dir is not None:
                    path = self._checkpoint_dir / 'state.pth'
                    print('New best validation loss, saving checkpoint to ',
                          path)
                    torch.save(self.best_state, path)
            if (
                self._checkpoint_dir is not None and
                i % self._checkpointing_freq == 0
            ):
                path = self._checkpoint_dir / \
                    f'state{self.scheduler.last_epoch}.pth'
                print(f'Saving checkpoint after epoch {i} to {path}')
                torch.save(self.state_dict(), path)
            print('-' * 100)

    def train_epoch(self):
        self.gan.train()
        loader = self.train_loader
        if self.logger is not None:
            from tqdm import tqdm
            loader = tqdm(loader)
        total_loss = 0
        for x, y in loader:
            loss = self.process_batch(x, y, train=True)
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.gan.eval()
        loader = self.val_loader
        if self.logger is not None:
            from tqdm import tqdm
            loader = tqdm(loader)
        total_loss = 0
        for x, y in loader:
            loss = self.process_batch(x, y, train=False)
            total_loss += loss.item()
        total_loss /= len(self.val_loader)
        if self.logger is not None:
            self.logger.commit()
            self.logger.run.log(
                {'val/loss': total_loss}, step=self.scheduler_g.last_epoch
            )
        return total_loss
