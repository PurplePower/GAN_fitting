import matplotlib.patches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from typing import Union

from visualizers.BaseSampler import BaseSampler

DEFAULT_SAVE_PATH = 'pics'


def make_line_points(n_samples: int, k: float = 1, b: float = 0, path=DEFAULT_SAVE_PATH):
    x = np.random.uniform(-1, 1, n_samples)
    y = k * x + b + np.random.normal(0, 0.25, n_samples)

    class _sampler(BaseSampler):

        def __init__(self):
            super().__init__(path, 'Single Line')
            self.figure = plt.figure()

        def call(self, x, epoch):
            plt.scatter(x[:, 0], x[:, 1])
            border = np.array([-2, 2])
            plt.plot(border, border * k + b, 'r')
            plt.title(f'Generated at epoch {epoch}')

    return np.array([x, y], dtype=np.float32).T, _sampler()


def make_cross_line_points(n_samples: int, k: float = 1, b: float = 0, path=DEFAULT_SAVE_PATH):
    """

    :param n_samples:
    :param k:
    :param b:
    :param path:
    :return: data matrix of shape (n_samples, data_dim)
    """
    x = np.random.uniform(-1, 1, n_samples)
    y = np.zeros_like(x)
    y[:n_samples // 2] = k * x[:n_samples // 2] + b
    y[n_samples // 2:] = -k * x[n_samples // 2:] + b
    scale = 3 / 40 * k + (1 / 40)
    y += np.random.normal(0, scale, y.shape)

    class _sampler(BaseSampler):
        def __init__(self):
            super().__init__(path, 'Cross-line')
            self.figure = plt.figure()

        def call(self, x, epoch):
            plt.scatter(x[:, 0], x[:, 1])

            border = np.array([-2, 2])
            plt.plot(border, border * k + b, 'r')
            plt.plot(border, border * (-k) + b, 'r')

            plt.title(f'Generated at epoch {epoch}')

    return np.stack([x, y]).astype(np.float32).T, _sampler()


def make_single_blob_points(n_samples: int, path=DEFAULT_SAVE_PATH):
    xy, label, centers = make_blobs(n_samples, 2, centers=[[3, 4]], return_centers=True)
    x, y = xy[:, 0], xy[:, 1]

    left, down = np.min(x), np.min(y)
    width, height = np.max(x) - left, np.max(y) - down

    class _sampler(BaseSampler):
        def __init__(self):
            super().__init__(path, 'Single Blob')
            self.figure, self.axes = plt.subplots()

        def call(self, x, epoch):
            fig, ax = self.figure, self.axes
            rect = matplotlib.patches.Rectangle((left, down), width, height, edgecolor='r', facecolor='none')
            ax.scatter(x[:, 0], x[:, 1])
            ax.scatter(centers[:, 0], centers[:, 1])
            ax.add_patch(rect)

            plt.title(f'Generated at epoch {epoch}')

    return np.array([x, y], dtype=np.float32).T, _sampler()


def make_ring_dots(n_samples: Union[int, list, np.ndarray], radius=2, path=DEFAULT_SAVE_PATH):
    centers = np.array([
        (radius * np.cos(theta := 2 * np.pi * (i / 8)), radius * np.sin(theta))
        for i in range(8)
    ])
    x, _ = make_blobs(n_samples, 2, centers=centers, cluster_std=0.1)

    class _sampler(BaseSampler):
        def __init__(self):
            super().__init__(path, 'Eight ring dots')
            self.figure = plt.figure()

        def call(self, xx, epoch):
            plt.scatter(xx[:, 0], xx[:, 1], 10)
            plt.scatter(centers[:, 0], centers[:, 1], marker='*')

            plt.title(f'Generated at epoch {epoch}')

    return np.array(x, dtype=np.float32), _sampler()
