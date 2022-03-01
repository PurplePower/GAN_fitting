"""
Base sampler. Inherit to sample an image every a few
training epochs.

"""

import abc
from pathlib import Path
import matplotlib.pyplot as plt


class BaseSampler(abc.ABC):

    def __init__(self, path=None, name=''):
        self.path = path    # if None, figure not saved
        self.name = name
        self.figure = None
        self.axes = None

        Path(path).mkdir(parents=True, exist_ok=True)  # ensure path exists
        pass

    def __call__(self, x, epoch):
        plt.ioff()
        plt.figure(self.figure.number)   # set current figure
        plt.clf()
        self.call(x, epoch)
        if self.path is not None:
            plt.savefig(f'{self.path}/d_{epoch}.png')
        pass

    @abc.abstractmethod
    def call(self, x, epoch):
        pass

    def set_path(self, p):
        self.path = p
