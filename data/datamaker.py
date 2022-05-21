import matplotlib.patches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_swiss_roll
from typing import Union
from PIL import Image

from visualizers.BaseSampler import BaseSampler
from visualizers.ScatterSampler import ScatterSampler
from utils.picdist import image2dots

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


def make_mog(
        n_samples: Union[int, list, np.ndarray], centers, cluster_std=.1, path=DEFAULT_SAVE_PATH,
        sample_scatter_kwargs=None, center_scatter_kwargs=None
):
    x, _ = make_blobs(n_samples, 2, centers=centers, cluster_std=cluster_std)

    # default sample points size 10
    sample_scatter_kwargs = sample_scatter_kwargs or dict()
    sample_scatter_kwargs['s'] = sample_scatter_kwargs.get('s', 10)

    # default center marker is *
    center_scatter_kwargs = center_scatter_kwargs or dict()
    center_scatter_kwargs['marker'] = center_scatter_kwargs.get('marker', '*')

    class _sampler(BaseSampler):
        def __init__(self):
            super().__init__(path, 'abstract mog')
            self.figure = plt.figure()
            self.centers = centers

        def call(self, xx, epoch):
            plt.scatter(xx[:, 0], xx[:, 1], **sample_scatter_kwargs)
            plt.scatter(self.centers[:, 0], self.centers[:, 1], **center_scatter_kwargs)

            plt.title(f'Generated at epoch {epoch}')

    return np.array(x, dtype=np.float32), _sampler()


def make_ring_dots(n_samples: Union[int, list, np.ndarray], radius=2.0, path=DEFAULT_SAVE_PATH):
    """
    8-mode mog
    """
    centers = np.array([
        (radius * np.cos(theta := 2 * np.pi * (i / 8)), radius * np.sin(theta))
        for i in range(8)
    ])
    # x, _ = make_blobs(n_samples, 2, centers=centers, cluster_std=0.1)
    #
    # class _sampler(BaseSampler):
    #     def __init__(self):
    #         super().__init__(path, 'Eight ring dots')
    #         self.figure = plt.figure()
    #         self.centers = centers
    #
    #     def call(self, xx, epoch):
    #         plt.scatter(xx[:, 0], xx[:, 1], 10)
    #         plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='*')
    #
    #         plt.title(f'Generated at epoch {epoch}')

    x, sampler = make_mog(n_samples, centers, path=path)
    sampler.name = 'Eight Mode mog'
    return x, sampler


def make_25_mog(n_samples: Union[int, list, np.ndarray], std=0.07, width=4.0, path=DEFAULT_SAVE_PATH):
    # make 5 * 5 grid centers
    xx = np.mgrid[-(width / 2):width / 2:5j]
    centers = np.array([
        (i, j)
        for j in xx for i in xx
    ])

    return make_mog(n_samples, centers, std, path)


def make_single_circle(n_samples: int, radius=2.0, std=0.1, path=DEFAULT_SAVE_PATH):
    # generate dots around the circle
    deg_step = 1.0
    points = np.zeros((n_samples, 2))

    rads = [i * deg_step * (np.pi / 180) for i in range(n_samples)]
    points = np.array([
        (radius * np.cos(rad), radius * np.sin(rad))
        for rad in rads
    ])
    points += np.random.normal(0, std, points.shape)  # add noise

    # TODO: use ScatterSampler
    class _sampler(BaseSampler):
        def __init__(self):
            super().__init__(path, 'Circle')
            self.figure = plt.figure()

        def call(self, x, epoch):
            plt.scatter(x[:, 0], x[:, 1], 10)
            plt.title(f'Generated at epoch {epoch}')

    return points.astype(np.float32), _sampler()


def make_sun(n_samples: int, radius_mog=2.0, radius_circle=1.0, path=DEFAULT_SAVE_PATH):
    n_mog = n_samples // 2
    n_circle = n_samples - n_mog

    mog, mog_sampler = make_ring_dots(n_mog, radius_mog, path)
    circle, circle_sampler = make_single_circle(n_circle, radius_circle, std=.1, path=path)
    plt.close(mog_sampler.figure)
    plt.close(circle_sampler.figure)

    points = np.concatenate([mog, circle], axis=0)

    class _sampler(BaseSampler):
        def __init__(self):
            super().__init__(path, 'Circle and mog')
            self.figure = plt.figure()
            self.centers = mog_sampler.centers

        def call(self, x, epoch):
            plt.scatter(x[:, 0], x[:, 1], 10)
            plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='*')

            plt.title(f'Generated at epoch {epoch}')

    return points.astype(np.float32), _sampler()


def make_from_image(n_samples: int, impath, path=DEFAULT_SAVE_PATH, canvas_len=4.0):
    """
    Load an image as gray scale and binarize it. Return points that are above the threshold.
    :param n_samples:
    :param impath:
    :param path:
    :param canvas_len:
    :return:
    """
    im = Image.open(impath)
    im = im.resize((64, 64))
    x = image2dots(im, n_samples, canvas_width=canvas_len, canvas_height=canvas_len)

    sampler = ScatterSampler(path, 'Image')
    sampler.xlim = sampler.ylim = (-(canvas_len / 2 * 1.1), canvas_len / 2 * 1.1)

    return x, sampler


def make_swiss_roll_2d(n_samples: int, noise=0.1, width=4.0, path=DEFAULT_SAVE_PATH):
    x, _ = make_swiss_roll(n_samples, noise=noise)  # 3D points

    x = x[:, [0, 2]]  # only x and z

    # centering
    shift = (np.max(x[:, 0]) + np.min(x[:, 0])) / 2
    x[:, 0] -= shift

    shift = (np.max(x[:, 1]) + np.min(x[:, 1])) / 2
    x[:, 1] -= shift

    # scale to canvas
    x[:, 0] *= width / (np.max(x[:, 0]) - np.min(x[:, 0]))
    x[:, 1] *= width / (np.max(x[:, 1]) - np.min(x[:, 1]))

    sampler = ScatterSampler(path, 'Swiss Roll 2D')
    sampler.xlim = sampler.ylim = (-(width / 2 * 1.1), width / 2 * 1.1)

    return x.astype(np.float32), sampler
