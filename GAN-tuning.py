import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
import numpy as np
import matplotlib.pyplot as plt
from models import GAN, WGAN, LSGAN, KernelGAN, fGAN, SWG, SWGAN
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from metrics.JudgeSurface import JudgeSurface
from utils.common import *
from utils.structures import *
from visualizers.plots import plot_2d_density, plot_2d_discriminator_judge_area

if __name__ == '__main__':
    model_type = SWGAN
    save_path = f'pics/{model_type.__name__.lower()}'
    empty_directory(save_path)
    # x, sampler = make_cross_line_points(1024, k=3, path=save_path)

    # make biased cluster size
    n_samples = 1024
    # biased_mog = False
    # if biased_mog:
    #     ratios = np.array([1, 2, 4, 8, 16, 24, 32, 40])
    #     cluster_sizes = ratios * n_samples / np.sum(ratios)
    #     cluster_sizes = np.around(cluster_sizes).astype(dtype=np.int32)
    #     if np.sum(cluster_sizes) < n_samples:
    #         cluster_sizes[0] += n_samples - np.sum(cluster_sizes)
    #     elif np.sum(cluster_sizes) > n_samples:
    #         cluster_sizes[-1] -= np.sum(cluster_sizes) - n_samples
    #     x, sampler = make_ring_dots(cluster_sizes, radius=2, path=save_path)
    # else:
    #     x, sampler = make_ring_dots(n_samples, radius=2, path=save_path)

    x, sampler = make_25_mog(n_samples, path=save_path)

    # x, sampler = make_sun(n_samples, path=save_path)

    sampler.formats = ['png', 'svg']

    # show original distribution
    plt.scatter(x[:, 0], x[:, 1])
    plt.savefig(f'{save_path}/dist.png')
    plt.clf()
    plt.hist2d(x[:, 0], x[:, 1], bins=100)
    plt.colorbar()
    plt.savefig(f'{save_path}/dist_density.png')
    plt.savefig(f'{save_path}/dist_density.svg')
    # plt.show()

    # dataset = tf.data.Dataset.from_tensor_slices(x)

    latent_factor = 5
    D, G = level_3a_structure(2, latent_factor)
    gan = None

    if model_type is GAN:
        gan = model_type(
            2, latent_factor=latent_factor, D=D, G=G,
            d_optimizer=SGD(1e-3), g_optimizer=SGD(1e-3))
    elif model_type is LSGAN:
        gan = model_type(
            2, latent_factor=latent_factor, D=D, G=G,
            d_optimizer=Adam(1e-3), g_optimizer=Adam(1e-3)
        )
    elif model_type is WGAN:
        gan = model_type(
            2, latent_factor=latent_factor, D=D, G=G,
            d_optimizer=SGD(1e-2), g_optimizer=SGD(1e-2)
            # d_optimizer=RMSprop(1e-3), g_optimizer=RMSprop(1e-3)
        )
    elif model_type is KernelGAN:
        bw_updater = PiecewiseConstantDecay([1000, 5000, 8000], [.5, .25, .125, .0625])
        # bw_updater = ExponentialDecay(0.45, 10000, 0.2)
        gan = KernelGAN(
            2, latent_factor=latent_factor, D=None, G=G,
            # d_optimizer=Adadelta(1), g_optimizer=Adadelta(1),
            d_optimizer=RMSprop(), g_optimizer=RMSprop(),
            bandwidth_updater=bw_updater, bandwidth=0.5
        )
    elif model_type is fGAN:
        gan = fGAN(
            2, latent_factor=latent_factor, D=D, G=G,
            d_optimizer=Adadelta(0.1), g_optimizer=Adadelta(0.1),
            divergence='pearson-chi-square'
        )
    elif model_type is SWG:
        # lr = PiecewiseConstantDecay([5000, 7000], [1e-4, 5e-5, 1e-5])
        ipe = n_samples // 64  # scale lr
        gan = SWG(
            2, latent_factor, D=D, G=G,
            # d_optimizer=SGD(1e-0), g_optimizer=SGD(1e-0),
            # d_optimizer=Adam(1e-4), g_optimizer=Adam(1e-4),
            d_optimizer=Adam(PiecewiseConstantDecay([5000 * ipe, 7000 * ipe], [1e-4, 5e-5, 1e-5])),
            g_optimizer=Adam(PiecewiseConstantDecay([5000 * ipe, 7000 * ipe], [1e-4, 5e-5, 1e-5])),
            use_discriminator=True, n_directions=128,
        )
    elif model_type is SWGAN:
        ipe = n_samples // 64  # scale lr
        gan = SWGAN(
            2, latent_factor, D=D, G=G,
            # d_optimizer=Adam(1e-5), g_optimizer=Adam(1e-5),
            d_optimizer=Adam(PiecewiseConstantDecay([5000 * ipe, 7000 * ipe], [1e-4, 5e-5, 1e-5])),
            g_optimizer=Adam(PiecewiseConstantDecay([5000 * ipe, 7000 * ipe], [1e-4, 5e-5, 1e-5])),
            lambda1=1.0, lambda2=1.0
        )
    else:
        raise Exception('Illegal GAN type.')

    # tf.profiler.experimental.start('log')

    sample_interval = 50
    if isinstance(gan, (KernelGAN, fGAN, SWGAN)):
        losses, metrics = gan.train(
            x, 10000, batch_size=64, sample_interval=sample_interval,
            sampler=sampler, sample_number=512,
            metrics=[JSD()])
    else:
        jsf = JudgeSurface(save_path + '/surfaces')
        losses, metrics = gan.train(
            x, 10000, batch_size=64, sample_interval=sample_interval,
            sampler=sampler, sample_number=512, dg_train_ratio=1,
            metrics=[JSD(), jsf])

    # tf.profiler.experimental.stop()
    empty_directory(save_path + '/model')
    gan.save(save_path + '/model')

    plt.figure()
    plt.plot(losses)
    plt.title('Losses over training epochs')
    plt.legend(['D losses', 'G losses'])
    plt.savefig(f'{save_path}/losses.png')
    plt.savefig(f'{save_path}/losses.svg')

    plt.figure()
    plt.plot(metrics[0])
    plt.title('JSD')
    plt.savefig(f'{save_path}/jsd.png')
    plt.savefig(f'{save_path}/jsd.svg')

    plt.figure('2D Density')
    plot_2d_density(gan, 256 * 1024)
    plt.savefig(f'{save_path}/density.png')
    plt.savefig(f'{save_path}/density.svg')

    plt.figure()
    plot_2d_discriminator_judge_area(gan, x, projection='3d')
    plt.savefig(f'{save_path}/judge area.png')
    plt.savefig(f'{save_path}/judge area.svg')

    plt.show()
    pass
