import random
import torch
import numpy as np


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class RunningMean:
    def __init__(self, gamma):
        self.gamma = gamma
        self.count = 0
        self._value = None

    def update(self, value):
        value = value.detach().cpu()
        if value.ndim == 0:
            self._update(value)
        else:
            for _v in value:
                self._update(_v)

    def _update(self, value):
        self.count += 1
        if self._value is None:
            self._value = value
        else:
            w1 = self.gamma * (1 - self.gamma ** (self.count - 1))
            w2 = (1 - self.gamma)
            wt = w1 + w2
            w1 = w1 / wt
            w2 = w2 / wt
            self._value = w1 * self._value + w2 * value

    @property
    def value(self):
        if self._value is None:
            return 0
        return self._value * 1
