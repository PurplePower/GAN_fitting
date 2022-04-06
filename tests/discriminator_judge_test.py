import matplotlib.pyplot as plt
import numpy as np

from visualizers.plots import plot_2d_discriminator_judge_area
from models import *
from data.datamaker import make_ring_dots

if __name__ == '__main__':
    # load a model then draw 2d density

    path = '../pics/wgan/model'
    model = WGAN.load(path)

    n_samples = 256 * 1024
    x, sampler = make_ring_dots(1024, radius=2)

    sampler(model.generate(256 * 1024), model.trained_epoch)

    plt.figure()
    plot_2d_discriminator_judge_area(model, x, resolution=150)

    plt.show()
