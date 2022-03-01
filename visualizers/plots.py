from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from models import BaseGAN


def plot_2d_density(model: BaseGAN, n_samples=1024, seed=None, bins=100):
    points = model.generate(n_samples, seed)
    # min_x, max_x = min(points[:, 0]), max(points[:, 0])
    # min_y, max_y = min(points[:, 1]), max(points[:, 1])
    #
    # nbins = 1000
    # xi, yi = np.mgrid[min_x:max_x:nbins * 1j, min_y:max_y:nbins * 1j]
    # sample_points = np.stack([xi.flatten(), yi.flatten()], axis=0).T
    #
    # kde = KernelDensity(bandwidth=bandwidth)
    # kde.fit(points)
    # log_probs = kde.score_samples(sample_points)

    plt.hist2d(points[:, 0], points[:, 1], bins=bins)
    plt.colorbar()
    pass
