from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models import BaseGAN


def plot_2d_density(model: BaseGAN, n_samples=1024, seed=None, bins=100):
    points = model.generate(n_samples, seed)
    plt.hist2d(points[:, 0], points[:, 1], bins=bins)
    plt.colorbar()
    pass


def plot_2d_discriminator_judge_area(
        model: BaseGAN, x=None, boundaries=None, resolution=100, extend_offset=0.5, projection='2d'):
    """
    Given 2D GAN, plot discriminator's judge area with probability
    :param model:
    :param x: data used to get boundaries, or None
    :param boundaries:
    :param resolution: number of sample points is resolution**2
    :return:
    """
    if x is None and boundaries is None:
        raise Exception("One of 'x' or 'boundaries' must be provided")
    if x is not None:
        min_x, max_x = np.min(x[:, 0]), np.max(x[:, 0])
        min_y, max_y = np.min(x[:, 1]), np.max(x[:, 1])
        min_x -= extend_offset
        min_y -= extend_offset
        max_x += extend_offset
        max_y += extend_offset
    else:
        min_x, max_x, min_y, max_y = boundaries

    xv, yv = np.mgrid[min_x:max_x:resolution * 1j, min_y:max_y:resolution * 1j]
    sample_points = np.stack([xv.flatten(), yv.flatten()], axis=0).T
    if model.discriminator:
        probs = np.array(model.discriminator(sample_points))
    else:
        probs = np.array(model.estimate_prob(x, sample_points))
    probs = probs.reshape(xv.shape)

    if projection == '3d':
        ax = plt.gca(projection='3d')
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(probs)
        ax.plot_surface(xv, yv, probs, cmap=m.cmap, norm=m.norm)

        plt.colorbar(m)
    else:
        cs = plt.contourf(xv, yv, probs)
        plt.colorbar(cs)
        plt.clabel(cs, inline=False, fontsize=10, colors='k')
    pass
