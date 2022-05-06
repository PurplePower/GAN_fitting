import abc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
import numpy as np
import itertools
from pathlib import Path

from models import BaseGAN, WGAN, KernelGAN, SWG
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from utils.common import empty_directory, get_optimizer_string
from utils.structures import *
from utils.runtests import *
from visualizers.plots import plot_2d_density, plot_2d_discriminator_judge_area


class WGANGetter(ModelGetter):
    def __init__(self, input_dim, latent_factor, lr_d, lr_g, **kwargs):
        super().__init__(input_dim, latent_factor, lr_d, lr_g)

    def get(self, structure, *args, **kwargs) -> WGAN:
        D, G = structure(self.input_dim, self.latent_factor)
        return WGAN(
            self.input_dim, self.latent_factor, D=D, G=G,
            d_optimizer=SGD(self.lr_d), g_optimizer=SGD(self.lr_g)
        )


class KernelGANGetter(ModelGetter):
    def __init__(self, input_dim, latent_factor, lr_d, lr_g,
                 bandwidth, bw_updater, **kwargs):
        super().__init__(input_dim, latent_factor, lr_d, lr_g)
        self.bandwidth = bandwidth
        self.bw_updater = bw_updater

    def get(self, structure, *args, **kwargs):
        D, G = structure(self.input_dim, self.latent_factor)
        return KernelGAN(
            self.input_dim, self.latent_factor, D=None, G=G,
            bandwidth=self.bandwidth, bandwidth_updater=self.bw_updater,
            d_optimizer=Adadelta(1), g_optimizer=Adadelta(1)
        )


class SWGGetter(ModelGetter):
    def __init__(
            self, input_dim, latent_factor, lr_d, lr_g, n_directions, lambda1, **kwargs):
        super().__init__(input_dim, latent_factor, lr_d, lr_g)
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


def get_folder_name(model: BaseGAN, batch_sz, structure):
    name = f'{model.__class__.__name__}'
    if model.d_optimizer:
        name += f'_dopt={get_optimizer_string(model.d_optimizer)}'

    name += f'_gopt={get_optimizer_string(model.g_optimizer)}'
    name += f'_bs={batch_sz}_lf={model.latent_factor}_struct={structure}'
    return name


if __name__ == '__main__':
    parent_path = Path('../htest/advanced')

    # basic model params
    input_dim = 2
    latent_factor = 5
    dg_r = 1
    epochs = 10000
    batch_size = 64
    structures = [level_3_structure, level_3a_structure]

    # model getters
    updater1 = CustomStairBandwidth([(2500, 0.5), (4000, 0.25), (6500, 0.125)], 0.08)
    getters = [
        WGANGetter(input_dim, latent_factor, 1e-3, 1e-3),
        WGANGetter(input_dim, latent_factor, 1e-2, 1e-2),
        KernelGANGetter(input_dim, latent_factor, 0, 0, 0.5, updater1),
        SWGGetter(input_dim, latent_factor, 1e-4, 1e-4, 100, 1.0)
    ]

    # training params
    repeat_times = 2
    sample_interval = 20

    n_samples = 1024
    ratios = np.array([1, 2, 4, 8, 16, 24, 32, 40])
    cluster_sizes = ratios * n_samples / np.sum(ratios)
    cluster_sizes = np.around(cluster_sizes).astype(dtype=np.int32)
    if np.sum(cluster_sizes) < n_samples:
        cluster_sizes[0] += n_samples - np.sum(cluster_sizes)
    elif np.sum(cluster_sizes) > n_samples:
        cluster_sizes[-1] -= np.sum(cluster_sizes) - n_samples
    x, sampler = make_ring_dots(cluster_sizes, radius=2)
    sampler.formats = ['png', 'svg']


    def savefig_in_fmts(path, basename):
        for fmt in sampler.formats:
            plt.savefig(f'{path}/{basename}.{fmt}')


    for model_getter, structure in itertools.product(getters, structures):
        model = model_getter.get(structure)

        folder = parent_path / get_folder_name(model, batch_size, STRUCTURE_NAMES[structure])
        folder.mkdir(parents=True, exist_ok=True)

        print(f'Training {model.name} in folder: {folder}')

        info: Info = Info.from_model(model, STRUCTURE_NAMES[structure], batch_size, 1)

        # write undone to dir
        info.done = False
        info.save(folder / 'config.json')

        # start training cases
        cases_to_run = get_cases_to_run(folder, repeat_times)
        print(f'cases to run: {cases_to_run}')
        for case in cases_to_run:
            print(f'Running case {case}...')
            model = model_getter.get(structure)
            case_path = folder / f'case-{case}'
            case_path.mkdir(exist_ok=True)
            empty_directory(case_path)  # clean before run

            # write case undone
            info.done = False
            info.save(case_path / 'config.json')

            # train
            sampler.set_path(case_path)
            losses, metrics = model.train(
                x, epochs, batch_size=batch_size, sample_interval=sample_interval, sample_number=512,
                sampler=sampler, metrics=[JSD()]
            )

            model.save(case_path / 'model')

            # plots
            plt.ioff()

            plt.clf()
            plt.plot(losses)
            plt.title('Losses over training epochs')
            plt.legend(['D losses', 'G losses'])
            savefig_in_fmts(case_path, 'losses')

            plt.clf()
            plt.plot(metrics[0])
            plt.title('Jensen-Shannon Divergence')
            plt.legend(['JSD'])
            savefig_in_fmts(case_path, 'JSD')

            plt.clf()
            plot_2d_density(model, 16 * 1024)
            plt.title('2D Density')
            savefig_in_fmts(case_path, 'Density 2D')

            plt.clf()
            plot_2d_discriminator_judge_area(model, x)
            plt.title('Discriminator judge area')
            savefig_in_fmts(case_path, 'Discriminator judge area')

            plt.clf()
            plot_2d_discriminator_judge_area(model, x, projection='3d')
            plt.title('Discriminator judge area')
            savefig_in_fmts(case_path, 'Discriminator judge area 3D')

            # write case done
            info.done = True
            info.save(case_path / 'config.json')

            pass

        # write done
        info.done = True
        info.save(folder / 'config.json')
