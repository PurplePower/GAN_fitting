"""
Fit 25-mog, swiss roll or more datasets.

"""

import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple
import json

from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from visualizers.BaseSampler import BaseSampler
from models import *
from utils.runtests import *
from utils.common import *
from utils.structures import *
from visualizers.plots import *


class SWGGetter(ModelGetter):
    def __init__(
            self, input_dim, latent_factor, lr_d, lr_g, n_directions, lambda1, tp, **kwargs):
        super().__init__(input_dim, latent_factor, lr_d, lr_g, tp)
        self.n_directions = n_directions
        self.lambda1 = lambda1

    def get(self, structure, *args, **kwargs) -> SWG:
        D, G = structure(self.input_dim, self.latent_factor)
        return SWG(
            self.input_dim, self.latent_factor, D=D, G=G,
            d_optimizer=Adam(self.lr_d), g_optimizer=Adam(self.lr_g),
            use_discriminator=True, n_directions=self.n_directions,
            lambda1=self.lambda1
        )


class WGANGetter(ModelGetter):
    def __init__(self, input_dim, latent_factor, lr_d, lr_g, tp, **kwargs):
        super().__init__(input_dim, latent_factor, lr_d, lr_g, tp)

    def get(self, structure, *args, **kwargs) -> WGAN:
        D, G = structure(self.input_dim, self.latent_factor)
        return WGAN(
            self.input_dim, self.latent_factor, D=D, G=G,
            d_optimizer=SGD(self.lr_d), g_optimizer=SGD(self.lr_g)
        )


class KernelGANGetter(ModelGetter):
    def __init__(self, input_dim, latent_factor, lr_d, lr_g,
                 bandwidth, bw_updater, tp, **kwargs):
        super().__init__(input_dim, latent_factor, lr_d, lr_g, tp)
        self.bandwidth = bandwidth
        self.bw_updater = bw_updater

    def get(self, structure, *args, **kwargs):
        D, G = structure(self.input_dim, self.latent_factor)
        return KernelGAN(
            self.input_dim, self.latent_factor, D=None, G=G,
            bandwidth=self.bandwidth, bandwidth_updater=self.bw_updater,
            d_optimizer=Adadelta(1), g_optimizer=Adadelta(1)
        )


def savefig_in_fmts(path, basename):
    for fmt in ['png', 'svg']:
        plt.savefig(f'{path}/{basename}.{fmt}')


def plot_data(x, path):
    plt.ioff()

    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], 10)
    plt.title('Data Distribution')
    savefig_in_fmts(path, 'data')


def plot_bundles(x, model, losses, metrics, path, **kwargs):
    # plots
    plt.ioff()

    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], 10)
    plt.title('Data Distribution')
    savefig_in_fmts(path, 'data')

    plt.clf()
    plt.plot(losses)
    plt.title('Losses over training epochs')
    plt.legend(['D losses', 'G losses'])
    savefig_in_fmts(path, 'losses')

    plt.clf()
    plt.plot(metrics[0])
    plt.title('Jensen-Shannon Divergence')
    plt.legend(['JSD'])
    savefig_in_fmts(path, 'JSD')

    plt.clf()
    plot_2d_density(model, 16 * 1024)
    plt.title('2D Density')
    savefig_in_fmts(path, 'Density 2D')

    plt.clf()
    plot_2d_discriminator_judge_area(model, x)
    plt.title('Discriminator judge area')
    savefig_in_fmts(path, 'Discriminator judge area')

    plt.clf()
    plot_2d_discriminator_judge_area(model, x, projection='3d')
    plt.title('Discriminator judge area')
    savefig_in_fmts(path, 'Discriminator judge area 3D')


def train_model_on_dataset(
        x, sampler: BaseSampler, model_getter: ModelGetter, struct, path, **kwargs
):
    path.mkdir(exist_ok=True)
    sampler.set_path(path)
    model = model_getter.get(struct)
    tp = model_getter.get_params()

    try:
        info = Info.load(path / 'info.json')
    except FileNotFoundError:
        info = Info.from_model(model, struct, tp['batch_size'])

    if info.done:
        print(f'Training already done, skipped.')
        return

    empty_directory(path)
    info.save(path / 'info.json')

    plot_data(x, path)

    losses, metrics = model.train(
        x, sampler=sampler, metrics=[JSD()], **tp
    )

    model.save(path / 'model')

    plot_bundles(x, model, losses, metrics, path)

    info.done = True
    info.save(path / 'info.json')

    pass


def fit_dataset(
        x, sampler: BaseSampler, model_and_params, struct, base_path: Path
):
    base_path.mkdir(parents=True, exist_ok=True)

    for t in model_and_params:
        path = base_path / t.name

        # empty_directory(path)

        print(f'Fitting dataset: {sampler.name} with model: {t.name}')
        train_model_on_dataset(x, sampler, t.getter, struct, path)

        print(f'Done with fitting dataset: {sampler.name} with model: {t.name}')


if __name__ == '__main__':
    save_path = Path('../htest/advanced')

    default_params = {
        'epochs': 10000,
        'batch_size': 64,
        'sample_interval': 50,
        'sample_number': 512
    }

    latent_factor = 5

    ##########################################
    #       swiss roll
    ##########################################
    getters = [
        WGANGetter(2, latent_factor, 1e-2, 1e-2, default_params | {'dg_train_ratio': 2}),
        KernelGANGetter(2, latent_factor, 0, 0, 0.5, PiecewiseConstantDecay([500, 1000, 2000], [.5, .25, .125, .125]),
                        default_params),
        SWGGetter(2, latent_factor, 1e-4, 1e-4, 100, 1, default_params)
    ]

    Params = namedtuple('Params', ['getter', 'name'])
    nts = [Params(g, n) for g, n in zip(getters, ['WGAN', 'KernelGAN', 'SWG'])]

    base_path = save_path / 'swiss roll'

    # make dataset
    x, sampler = make_swiss_roll_2d(1024, path=base_path)
    sampler.formats = ['png', 'svg']

    # special adjustments
    for g in getters:
        g.training_params['epochs'] = 3000

    # fitting
    fit_dataset(x, sampler, nts, level_3a_structure, base_path)

    ############################################
    #       25-mode mog
    ############################################
    base_path = save_path / '25-mode mog'

    # param adjust
    getters[0].training_params.update({'dg_train_ratio': 3, 'epochs': 9000})

    # make dataset
    x, sampler = make_25_mog(1024, path=base_path)

    # fitting
    fit_dataset(x, sampler, nts, level_3a_structure, base_path)

    # x, sampler = make_from_image(1024, '../pics/wgan_text.png')
    # fit_dataset(x, sampler, nts, level_4_structure, save_path / 'image')

    pass
