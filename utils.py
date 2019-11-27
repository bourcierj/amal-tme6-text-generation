"""Utilities for training: chekpointing, early stopping."""

import torch
from pathlib import Path

class CheckpointState():
    """A model checkpoint state."""
    def __init__(self, model, optimizer=None, epoch=1, savepath='./checkpt.pt'):

        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.savepath = Path(savepath)

    def state_dict(self):
        """Checkpoint's state dict, to save and load"""
        dict_ = dict()
        dict_['model'] = self.model.state_dict()
        if self.optimizer:
            dict_['optimizer'] = self.optimizer.state_dict()
        dict_['epoch'] = self.epoch
        return dict_

    def save(self, suffix=''):
        """Serializes the checkpoint.
        Args:
            suffix (str): if provided, a suffix will be prepended before the extension
                of the object's savepath attribute.
        """
        if suffix:
            savepath = self.savepath.parent / Path(self.savepath.stem + suffix +
                                                   self.savepath.suffix)
        else:
            savepath = self.savepath
        with savepath.open('wb') as fp:
            torch.save(self.state_dict(), fp)

    def load(self):
        """Deserializes and map the checkpoint to the available device."""
        with self.savepath.open('rb') as fp:
            state_dict = torch.load(
                fp, map_location=torch.device('cuda' if torch.cuda.is_available()
                                              else 'cpu'))
            self.update(state_dict)

    def update(self, state_dict):
        """Updates the object with a dictionary
        Args:
            state_dict (dict): a dictionary with keys:
                - 'model' containing a state dict for the checkpoint's model
                - 'optimizer' containing a state for the checkpoint's optimizer
                  (optional)
                - 'epoch' containing the associated epoch number
        """
        self.model.load_state_dict(state_dict['model'])
        if self.optimizer is not None and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']


class EarlyStopper():
    """Object to use early stopping during training"""
    def __init__(self, patience):

        self.patience = patience
        self.min_loss = float('inf')
        self.min_epoch = 1

    def add(self, loss, epoch):
        if loss <= self.min_loss:
            self.min_loss = loss
            self.min_epoch = epoch
        self.last_epoch = epoch

    def stop(self):
        return ((self.last_epoch - self.min_epoch) > self.patience)


if __name__ == '__main__':

    import os
    import torch.nn as nn
    import torch.optim as optim

    # Test CheckpointState
    module = nn.Linear(1, 2)
    opt = optim.SGD(module.parameters(), lr=0.01)
    checkpt = CheckpointState(module, opt, savepath='./checkpt-test.pt')
    checkpt.save()
    checkpt.save('_best')
    module = nn.Linear(1, 2)
    opt = optim.SGD(module.parameters(), lr=0.01)
    checkpt2 = CheckpointState(module, opt, savepath='./checkpt-test.pt')
    checkpt2.load()
    print('Checkpoint 1 state dict:')
    print(checkpt.state_dict())
    print('Checkpoint 2 state dict:')
    print(checkpt2.state_dict())
    for p in Path('./').glob('checkpt-test*.pt'):
        os.remove(p)

    # Test EarlyStopper
    early_stopper = EarlyStopper(patience=10)
    values = list(range(20, 0, -1)) + list(range(0, 20))
    for epoch, v in enumerate(values, 1):
        early_stopper.add(v, epoch)
        if early_stopper.stop():
            print('Early stopping at epoch ', epoch)
            break

    print('Min epoch:', early_stopper.min_epoch)
    print('Min loss:', early_stopper.min_loss)
