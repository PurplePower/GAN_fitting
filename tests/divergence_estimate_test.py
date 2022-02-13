import sklearn
from sklearn.neighbors import KernelDensity
import numpy as np
from data.datamaker import *
import matplotlib.pyplot as plt

RADIUS = 2


def get_densities(X, bandwidth, mg_resolution):
    xv, yv = np.meshgrid(
        np.linspace(-1.1 * RADIUS, 1.1 * RADIUS, mg_resolution),
        np.linspace(-1.1 * RADIUS, 1.1 * RADIUS, mg_resolution))
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X)
    tv = np.stack([xv.flatten(), yv.flatten()], axis=0).T
    densities = kde.score_samples(tv)  # log probs
    densities = np.exp(densities.reshape((int(densities.size ** 0.5), -1)))
    return densities / np.sum(densities.flatten()), xv, yv


def kl(p, q):
    return np.sum(
        p * (np.log(p / q))
    )


def jsd(p, q):
    m = (p + q) / 2
    return (kl(p, m) + kl(q, m)) / 2


if __name__ == '__main__':
    X, _ = make_ring_dots(1024, radius=RADIUS)

    plt.scatter(X[:, 0], X[:, 1])
    plt.title('original data distribution')

    bw = RADIUS / 10
    densities, xv, yv = get_densities(X, bw, 50)

    plt.figure()
    plt.contourf(densities)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(xv, yv, densities)

    print(f'JS-div of X to itself = {jsd(densities, densities)} in bandwidth={bw}')

    # shift X
    shift_X = X + 0.1
    shifted_densities, _, _ = get_densities(shift_X, bw, 50)
    print(f'JS-div of X to shifted = {jsd(densities, shifted_densities)} in bandwidth={bw}')

    plt.show()
    pass
