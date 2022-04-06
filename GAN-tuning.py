import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import matplotlib.pyplot as plt
from models import WGAN, KernelGAN, fGAN
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from utils.common import *
from utils.structures import *
from visualizers.plots import plot_2d_density, plot_2d_discriminator_judge_area

if __name__ == '__main__':
    model_type = fGAN
    save_path = f'pics/{model_type.__name__.lower()}'
    clean_images(save_path)
    # x, sampler = make_cross_line_points(1024, k=3, path=save_path)

    # make biased cluster size
    n_samples = 1024
    biased_mog = False
    if biased_mog:
        ratios = np.array([1, 2, 4, 8, 16, 24, 32, 40])
        cluster_sizes = ratios * n_samples / np.sum(ratios)
        cluster_sizes = np.around(cluster_sizes).astype(dtype=np.int32)
        if np.sum(cluster_sizes) < n_samples:
            cluster_sizes[0] += n_samples - np.sum(cluster_sizes)
        elif np.sum(cluster_sizes) > n_samples:
            cluster_sizes[-1] -= np.sum(cluster_sizes) - n_samples
        x, sampler = make_ring_dots(cluster_sizes, radius=2, path=save_path)
    else:
        x, sampler = make_ring_dots(n_samples, radius=2, path=save_path)

    # show original distribution
    plt.scatter(x[:, 0], x[:, 1])
    plt.savefig(f'{save_path}/dist.png')
    plt.clf()
    plt.hist2d(x[:, 0], x[:, 1], bins=100)
    plt.colorbar()
    plt.savefig(f'{save_path}/dist_density.png')
    # plt.show()

    # dataset = tf.data.Dataset.from_tensor_slices(x)

    latent_factor = 5
    D, G = level_1_structure(2, latent_factor)


    def updater(bw, epoch, epochs):
        if epoch < 2500:
            return 0.5
        elif epoch < 4000:
            return 0.25
        elif epoch < 6500:
            return 0.125
        else:
            return 0.08


    # gan = model_type(
    #     2, latent_factor=latent_factor, D=None, G=G,
    #     bandwidth=0.5, bandwidth_updater=updater,
    #     d_optimizer=Adadelta(1), g_optimizer=Adadelta(1))
    gan = model_type(
        2, latent_factor=latent_factor, D=None, G=G,
        d_optimizer=Adadelta(0.1),
        g_optimizer=Adadelta(0.1),
    # )
        divergence='squared-hellinger')
    # , d_optimizer=SGD(1e-3), g_optimizer=SGD(1e-3)

    # tf.profiler.experimental.start('log')

    losses, metrics = gan.train(
        x, 2000, batch_size=64, sample_interval=20,
        sampler=sampler, sample_number=512,
        metrics=[JSD()])

    # tf.profiler.experimental.stop()
    empty_directory(save_path + '/model')
    gan.save(save_path + '/model')

    plt.figure()
    plt.plot(losses)
    plt.title('Losses over training epochs')
    plt.legend(['D losses', 'G losses'])
    plt.savefig(f'{save_path}/losses.png')

    plt.figure()
    plt.plot(metrics[0])
    plt.title('JSD')
    plt.savefig(f'{save_path}/jsd.png')

    plt.figure('2D Density')
    plot_2d_density(gan, 256 * 1024)
    plt.savefig(f'{save_path}/density')

    plt.figure()
    plot_2d_discriminator_judge_area(gan, x, projection='3d')

    plt.show()
    pass
