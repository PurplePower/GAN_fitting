import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
from models import GAN, WGAN, LSGAN, GAN_Tensor
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from utils.common import *
from utils.structures import *
from visualizers.plots import plot_2d_density

if __name__ == '__main__':
    model_type = WGAN
    save_path = f'pics/{model_type.__name__.lower()}'
    clean_images(save_path)
    # x, sampler = make_cross_line_points(1024, k=3, path=save_path)

    # make biased cluster size
    n_samples = 1024
    ratios = np.array([1, 2, 4, 8, 16, 24, 32, 40])
    cluster_sizes = ratios * n_samples / np.sum(ratios)
    cluster_sizes = np.around(cluster_sizes).astype(dtype=np.int32)
    if np.sum(cluster_sizes) < n_samples:
        cluster_sizes[0] += n_samples - np.sum(cluster_sizes)
    elif np.sum(cluster_sizes) > n_samples:
        cluster_sizes[-1] -= np.sum(cluster_sizes) - n_samples

    x, sampler = make_ring_dots(cluster_sizes, radius=2, path=save_path)

    # show original distribution
    plt.scatter(x[:, 0], x[:, 1])
    plt.savefig(f'{save_path}/dist.png')
    # plt.show()

    dataset = tf.data.Dataset.from_tensor_slices(x)

    latent_factor = 5
    D, G = level_3a_structure(2, latent_factor)
    gan = model_type(2, latent_factor=latent_factor, D=D, G=G)
    # , d_optimizer=SGD(1e-3), g_optimizer=SGD(1e-3)

    # tf.profiler.experimental.start('log')

    losses, metrics = gan.train(x, 10000, batch_size=64, sample_interval=20,
                                sampler=sampler, sample_number=512,
                                dg_train_ratio=1, metrics=[JSD()])

    # tf.profiler.experimental.stop()
    empty_directory(save_path + '/model')
    gan.save(save_path + '/model')

    plt.figure()
    plt.plot(losses)
    plt.title('Losses over training epochs')
    plt.legend(['D losses', 'G losses'])
    plt.savefig('losses.png')

    plt.figure()
    plt.plot(metrics[0])
    plt.title('JSD')
    plt.savefig('jsd.png')

    plt.figure('2D Density')
    plot_2d_density(gan, 256*1024)
    plt.savefig('density')

    plt.show()
    pass
