import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import itertools
from pathlib import Path

from models import SWG, BaseGAN
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from utils.common import empty_directory, round_to_fit
from utils.structures import *
from utils.runtests import *
from visualizers.plots import *


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


def get_folder_name(model: SWG, batch_sz, structure):
    name = f'SWG_dopt=Adam({round_to_fit(model.d_optimizer.learning_rate.numpy())})_'
    name += f'gopt=Adam({round_to_fit(model.g_optimizer.learning_rate.numpy())})_'
    name += f'bs={batch_sz}_lf={model.latent_factor}_struct={structure}_'
    name += f'ndir={model.n_directions}_lambda={model.lambda1}'
    return name


if __name__ == '__main__':
    parent_path = Path('htest/gp')

    # basic model params
    input_dim = 2
    latent_factor = 5
    epochs = 10000
    batch_size = 64

    # variables
    # structures = [level_3_structure]
    structures = [level_1_structure, level_2_structure]

    """
    Extremely unstable to use structure 3 with 1e-3 lr
    """
    learning_rates = [(1e-4, 1e-4)]
    n_directions = [100]
    lambdas = [1.0]

    # make getters
    getters = [
        SWGGetter(input_dim, latent_factor, lr_d, lr_g, n_directions=n_dir, lambda1=lmd)
        for (lr_d, lr_g), n_dir, lmd in itertools.product(learning_rates, n_directions, lambdas)
    ]

    # get data
    x, sampler = make_ring_dots(1024, path='')
    sampler.formats = ['png', 'svg']


    def savefig_in_fmts(path, basename):
        for fmt in sampler.formats:
            plt.savefig(f'{path}/{basename}.{fmt}')


    # training params
    repeats = 1
    sample_interval = 50

    # testing
    for getter, structure in itertools.product(getters, structures):
        model = getter.get(structure)

        folder = parent_path / get_folder_name(model, batch_size, STRUCTURE_NAMES[structure])
        folder.mkdir(parents=True, exist_ok=True)

        print(f'Training {model.name} in folder: {folder}')

        info: Info = Info.from_model(model, STRUCTURE_NAMES[structure], batch_size)

        # write undone
        info.done = False
        info.save(folder / 'config.json')

        # run cases
        cases_to_run = get_cases_to_run(folder, repeats)
        print(f'cases to run {cases_to_run}')
        for case in cases_to_run:
            print(f'Running case {case}...')
            model = getter.get(structure)
            case_path = folder / f'case-{case}'
            case_path.mkdir(exist_ok=True)
            empty_directory(case_path)  # clean before run

            # write case undone
            info.done = False
            info.save(case_path / 'config.json')

            # train
            sampler.set_path(case_path)
            losses, metrics = model.train(
                x, epochs, batch_size=batch_size, sample_interval=sample_interval,
                sample_number=512, sampler=sampler,
                metrics=[JSD()]
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

    pass
