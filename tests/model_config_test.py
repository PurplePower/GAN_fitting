import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from models import *

from data.datamaker import make_ring_dots
from utils.structures import level_1_structure

input_dim = 2
latent_factor = 5


def gan_getter():
    model = GAN(input_dim, latent_factor)
    return model


def wgan_getter():
    return WGAN(
        input_dim, latent_factor,
        d_optimizer=SGD(ExponentialDecay(0.01, 10, 0.7)))


def lsgan_getter():
    return LSGAN(input_dim, latent_factor, g_optimizer=Adadelta(1))


def fgan_getter():
    return fGAN(input_dim, latent_factor, g_optimizer=Adadelta(1))


def kernelgan_getter():
    return KernelGAN(
        input_dim, latent_factor,
        bandwidth=0.5,
        bandwidth_updater=CustomStairBandwidth([(3, 0.5)], 0.25)
    )


if __name__ == '__main__':
    getters = [gan_getter, wgan_getter, lsgan_getter, fgan_getter, kernelgan_getter]

    x, sampler = make_ring_dots(1024)

    for getter in getters:
        model: BaseGAN = getter()
        model.train(x, 5, 64)

        config = model.get_config()
        print(f'Model config of {model.__class__.__name__}:\n{config}')

        new_model = model.__class__.from_config(config)

        new_model.train(x, 5, 64)

        print(f'{model.__class__.__name__} from config success.')
        print(f'\n{"=" * 20}\n')
