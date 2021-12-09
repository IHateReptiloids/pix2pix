from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F


class L1Trainer:
    def __init__(
        self,
        config,
        model,
        opt,
        scheduler,
        train_loader,
        val_loader,
        logger=None
    ):
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.best_loss = None
        self.best_state = None

        self._checkpointing_freq = config.checkpointing_freq
        self._checkpoint_dir = None
        if self.logger is not None:
            self._checkpoint_dir = Path(self.logger.checkpoint_dir)

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.opt.load_state_dict(state['opt'])
        self.scheduler.load_state_dict(state['scheduler'])

    def process_batch(self, x, y, train: bool):
        out = self.model(x)
        loss = F.l1_loss(out, y)
        if self.logger is not None:
            self.logger.log_batch(x, y, out, loss.item(), train=train)
        return loss

    def state_dict(self):
        state = OrderedDict()
        state['model'] = self.model.state_dict()
        state['opt'] = self.opt.state_dict()
        state['scheduler'] = self.scheduler.state_dict()
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
        self.model.train()
        loader = self.train_loader
        if self.logger is not None:
            from tqdm import tqdm
            loader = tqdm(loader)
        total_loss = 0
        for x, y in loader:
            loss = self.process_batch(x, y, train=True)
            total_loss += loss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.scheduler.step()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        loader = self.val_loader
        if self.logger is not None:
            from tqdm import tqdm
            loader = tqdm(loader)
        total_loss = 0
        for x, y in loader:
            loss = self.process_batch(x, y, train=False)
            total_loss += loss.item()
        if self.logger is not None:
            self.logger.commit()
        return total_loss / len(self.val_loader)
