"""
Base sampler. Inherit to sample an image every a few
training epochs.

"""

import abc
from pathlib import Path
import matplotlib.pyplot as plt


class BaseSampler(abc.ABC):

    def __init__(self, path, name=''):
        self.path = path
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
        pass

    @abc.abstractmethod
    def call(self, x, epoch):
        pass

    def set_path(self, p):
        self.path = p
