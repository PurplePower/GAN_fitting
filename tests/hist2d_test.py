import matplotlib.pyplot as plt
import numpy as np

from visualizers.plots import plot_2d_density
from models import *
from data.datamaker import make_ring_dots

if __name__ == '__main__':
    # load a model then draw 2d density

    path = '../pics/wgan/model'
    model = WGAN.load(path)

    n_samples = 256 * 1024
    _, sampler = make_ring_dots(1024, radius=2)

    sampler(model.generate(256 * 1024), model.trained_epoch)

    plt.figure()
    plot_2d_density(model, 16 * 1024)

    plt.show()
