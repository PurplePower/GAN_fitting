import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from models import GAN, WGAN, LSGAN, GAN_Tensor
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from utils.common import clean_images

if __name__ == '__main__':
    model_type = GAN_Tensor
    save_path = f'pics/{model_type.__name__.lower()}'
    clean_images(save_path)
    # x, sampler = make_cross_line_points(1024, k=3, path=save_path)
    x, sampler = make_ring_dots(1024, radius=2, path=save_path)

    # show original distribution
    plt.scatter(x[:, 0], x[:, 1])
    plt.savefig(f'{save_path}/dist.png')
    # plt.show()

    dataset = tf.data.Dataset.from_tensor_slices(x)

    SGD = keras.optimizers.SGD
    gan = model_type(2, latent_factor=5)
    # , d_optimizer=SGD(1e-3), g_optimizer=SGD(1e-3)

    # tf.profiler.experimental.start('log')

    losses, metrics = gan.train(x, 40, batch_size=64, sample_interval=20, sampler=sampler,
                                dg_train_ratio=1, metrics=[JSD()])

    # tf.profiler.experimental.stop()

    plt.figure()
    plt.plot(losses)
    plt.title('Losses over training epochs')
    plt.legend(['D losses', 'G losses'])

    plt.figure()
    plt.plot(metrics[0])
    plt.title('JSD')

    plt.show()
    pass
