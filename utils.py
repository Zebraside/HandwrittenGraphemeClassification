from fastai.callbacks import TrackerCallback
from fastai.vision import Learner
from torch import nn
import torch.nn.functional as F
import os


class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."

    def __init__(self,
                 learn: Learner,
                 monitor: str = 'valid_loss',
                 mode: str = 'auto',
                 every: str = 'improvement',
                 name: str = 'bestmodel'):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every, self.name = every, name
        if self.every not in ['improvement', 'epoch']:
            warn(
                f'SaveModel every {self.every} is invalid, falling back to "improvement".'
            )
            self.every = 'improvement'

    def jump_to_epoch(self, epoch: int) -> None:
        try:
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except:
            print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every == "epoch":
            # self.learn.save(f'{self.name}_{epoch}')
            torch.save(learn.model.state_dict(), f'{self.name}_{epoch}.pth')
        else:  # every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                # print(f'Better model found at epoch {epoch} \
                #  with {self.monitor} value: {current}.')
                self.best = current
                # self.learn.save(f'{self.name}')
                torch.save(learn.model.state_dict(), f'{self.name}.pth')

    def on_train_end(self, **kwargs):
        "Load the best model."
        if self.every == "improvement" and os.path.isfile(f'{self.name}.pth'):
            # self.learn.load(f'{self.name}', purge=False)
            self.model.load_state_dict(torch.load(f'{self.name}.pth'))


class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        x1, x2, x3 = input
        y = target.long()
        return 2.0 * F.cross_entropy(x1, y[:, 0]) + \
            F.cross_entropy(x2, y[:, 1]) + \
            F.cross_entropy(x3, y[:, 2])
