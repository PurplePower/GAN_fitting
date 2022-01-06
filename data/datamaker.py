import matplotlib.patches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def make_line_points(n_samples: int, k: float = 1, b: float = 0, path='pics'):
    x = np.random.uniform(-1, 1, n_samples)
    y = k * x + b + np.random.normal(0, 0.25, n_samples)

    def image_saver(G, epoch, noise):
        points = G(noise, training=False)
        plt.clf()  # clear
        plt.scatter(points[:, 0], points[:, 1])
        border = np.array([-2, 2])
        plt.plot(border, border * k + b, 'r')
        plt.title(f'Generated at epoch {epoch}')
        plt.savefig(f'{path}/d_{epoch}.png')
        plt.close()

    return x, y, image_saver


def make_cross_line_points(n_samples: int, k: float = 1, b: float = 0, path='pics'):
    x1 = np.random.uniform(-1, 1, n_samples // 2)
    y1 = k * x1 + b + np.random.normal(0, 0.25, len(x1))

    x2 = np.random.uniform(-1, 1, n_samples // 2)
    y2 = -k * x2 + b + np.random.normal(0, 0.25, len(x2))

    def image_saver(G, epoch, noise):
        points = G(noise, training=False)
        plt.clf()  # clear
        plt.scatter(points[:, 0], points[:, 1])

        border = np.array([-2, 2])
        plt.plot(border, border * k + b, 'r')
        plt.plot(border, border * (-k) + b, 'r')

        plt.title(f'Generated at epoch {epoch}')
        plt.savefig(f'{path}/d_{epoch}.png')
        plt.close()

    return np.concatenate([x1, x2]), np.concatenate([y1, y2]), image_saver


def make_single_blob_points(n_samples: int, path='pics'):
    xy, label, centers = make_blobs(n_samples, 2, centers=[[3, 4]], return_centers=True)
    x, y = xy[:, 0], xy[:, 1]

    left, down = np.min(x), np.min(y)
    width, height = np.max(x) - left, np.max(y) - down

    def image_saver(G, epoch, noise):
        points = G(noise, training=False)
        fig, ax = plt.subplots()
        rect = matplotlib.patches.Rectangle((left, down), width, height, edgecolor='r', facecolor='none')
        ax.scatter(points[:, 0], points[:, 1])
        ax.scatter(centers[:, 0], centers[:, 1])
        ax.add_patch(rect)

        plt.title(f'Generated at epoch {epoch}')
        plt.savefig(f'{path}/d_{epoch}.png')
        plt.close(fig)

    return x, y, image_saver
