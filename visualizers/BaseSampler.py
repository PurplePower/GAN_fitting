"""
Base sampler. Inherit to sample an image every a few
training epochs.

"""

import abc
from pathlib import Path
import matplotlib.pyplot as plt


class BaseSampler(abc.ABC):

    def __init__(
            self, path=None, name='', formats='png',
            xlim=None, ylim=None
    ):
        """

        :param path:
        :param name:
        :param formats: picture extension. If list given, save multiples of each format.
        """
        self.path = path    # if None, figure not saved
        self.name = name
        self.figure = None
        self.axes = None
        self.formats = formats
        self.xlim, self.ylim = xlim, ylim
        plt.ioff()

        if path:
            Path(path).mkdir(parents=True, exist_ok=True)  # ensure path exists
        pass

    def __call__(self, x, epoch):
        plt.ioff()
        plt.figure(self.figure.number)   # set current figure
        plt.clf()
        self.call(x, epoch)
        if self.xlim:
            plt.xlim(self.xlim)
        if self.ylim:
            plt.ylim(self.ylim)
        if self.path is not None:
            if isinstance(self.formats, list):
                for f in self.formats:
                    plt.savefig(f'{self.path}/d_{epoch}.{f}')
            else:
                plt.savefig(f'{self.path}/d_{epoch}.png')
        pass

    @abc.abstractmethod
    def call(self, x, epoch):
        pass

    def set_path(self, p):
        self.path = p
