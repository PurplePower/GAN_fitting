import matplotlib.pyplot as plt

from visualizers.BaseSampler import BaseSampler


class ScatterSampler(BaseSampler):
    def __init__(
            self, path=None, name='', formats='png',
            marker_size=10, marker='.', **kwargs
    ):
        super(ScatterSampler, self).__init__(path, name, formats, **kwargs)
        self.figure = plt.figure()
        self.marker_size = marker_size
        self.marker = marker

    def call(self, x, epoch):
        plt.scatter(x[:, 0], x[:, 1], self.marker_size, marker=self.marker)
        plt.title(f'Generated at epoch {epoch}')

