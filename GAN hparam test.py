import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, LeakyReLU
import numpy as np
import itertools
from pathlib import Path
import json
import hashlib
import shutil

from models import GAN, BaseGAN
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from utils.common import empty_directory


class Info:
    def __init__(self, **kwargs):
        self.input_dim = 0
        self.latent_factor = 0
        self.lr_d = self.lr_g = 0
        self.opt = ''
        self.batch_sz = 0
        self.dg_r = 0
        self.struct = None
        self.done = False
        self.trained_epochs = 0
        self.__dict__.update(**kwargs)

    @classmethod
    def load(cls, path):
        i = Info()
        with open(path, 'r') as f:
            i.__dict__ = json.load(f)
        return i

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        pass


def level_1_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(32, input_shape=(input_dim,)), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(32, input_shape=(latent_factor,)), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_2_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(32, input_shape=(input_dim,)), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(32, input_shape=(latent_factor,)), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


STRUCTURE_NAMES = {
    level_1_structure: '1',
    level_2_structure: '2'
}


def folder_name_getter(i: Info):
    """
    GAN_lrd={lr_d}_lrg={lr_g}_bs={batch_size}_dgr={dg_r}_opt={optimizer_type}_lf={latent_factor}_strct={json hash:X}
    Use hashlib for stable hashing.
    :return:
    """

    name = f'GAN_lrd={i.lr_d}_lrg={i.lr_g}_bs={i.batch_sz}_dgr={i.dg_r}_opt={i.opt}' \
           f'_lf={i.latent_factor}_strct={i.struct}'
    return name


def run_single_case(path, case, clear_before_run=True, **kwargs):
    case_path = path / f'case-{case}'
    case_path.mkdir(exist_ok=True)
    if clear_before_run:
        empty_directory(case_path)

    # write undone case-config
    info = kwargs['info']
    info.done = False
    info.save(case_path / 'config.json')

    structure = kwargs['structure']
    model_params = kwargs['model_params']

    # build model
    D, G = structure(input_dim, latent_factor)
    model = GAN(D=D, G=G, **model_params)

    sampler = kwargs['sampler']
    sampler.set_path(case_path)

    training_params = kwargs['training_params']
    training_params['sampler'] = sampler
    losses, metrics = model.train(**training_params)

    # save the model
    model.save(case_path / 'model')

    # save figures
    plt.ioff()
    plt.clf()
    plt.plot(losses)
    plt.title('Losses over training epochs')
    plt.legend(['D losses', 'G losses'])
    plt.savefig(case_path / 'losses.png')

    plt.clf()
    plt.plot(metrics[0])
    plt.title('Jensen-Shannon Divergence')
    plt.legend(['JSD'])
    plt.savefig(case_path / 'JSD.png')

    # write done case-config
    info.done = True
    info.save(case_path / 'config.json')

    pass


if __name__ == '__main__':
    """
    test covers:
        learning rate combination
        model structure
        latent factor size
        alternative learning rounds
        (default) SGD optimizers
    """
    parent_path = Path('htest')

    # model hyper-params
    input_dim = 2
    default_opt = keras.optimizers.SGD
    structures = [level_1_structure, level_2_structure]

    # training params
    learning_rate_combinations = [(1e-3, 1e-3), (1e-2, 1e-2), (1e-3, 5e-3)]
    latent_factors = [5]
    dg_train_ratios = [1, 3]
    epochs = 10000
    batch_size = 64

    #
    repeat_times = 5  # cases per hparam
    sample_interval = 20

    x, sampler = make_ring_dots(1024, path='')  # one dataset for all trainings

    for (lr_d, lr_g), latent_factor, structure, dg_r in itertools.product(learning_rate_combinations, latent_factors,
                                                                          structures, dg_train_ratios):
        D, G = structure(input_dim, latent_factor)
        model = GAN(input_dim, latent_factor, D=D, G=G,
                    d_optimizer=default_opt(lr_d), g_optimizer=default_opt(lr_g))

        # pack training and model information
        info = Info(
            input_dim=input_dim, latent_factor=latent_factor, opt=default_opt.__name__,
            lr_d=lr_d, lr_g=lr_g, batch_sz=batch_size, dg_r=dg_r, struct=STRUCTURE_NAMES[structure],
            trained_epochs=epochs
        )

        folder_name = folder_name_getter(info)
        path = parent_path / folder_name
        cases_to_run = []

        if path.exists():
            try:
                existed_info = Info.load(path / 'config.json')
            except FileNotFoundError:
                print(f'Config not found , retrain all cases in path {path}')
                empty_directory(path)

                cases_to_run = list(range(repeat_times))
            else:
                trained_cases = set()
                for folder in sorted(path.glob('case*')):
                    s = str(folder)
                    case_num = int(s[s.find('-') + 1:])
                    try:
                        case_info = Info.load(folder / 'config.json')
                    except FileNotFoundError:
                        cases_to_run.append(case_num)
                    else:
                        if not case_info.done:
                            cases_to_run.append(case_num)
                        else:
                            trained_cases.add(case_num)
                    pass

                max_case = max(trained_cases) if trained_cases else -1
                assert len(trained_cases) == max_case + 1
                if existed_info.done and max_case + 1 == repeat_times:
                    print(f'Training all done, skipped in path {path}')
                    continue
                else:
                    cases_to_run.extend([
                        i for i in range(repeat_times) if i not in trained_cases and i not in cases_to_run
                    ])

        else:
            path.mkdir(parents=True)
            cases_to_run = list(range(repeat_times))  # run all cases
            pass

        print(f'Training cases {cases_to_run} will run in path {path}')

        # write undone config
        info.done = False
        info.save(path / 'config.json')

        # run cases
        for case in cases_to_run:
            print(f'Running case {case}...')
            kwargs = {  # build each case to build new optimizers and metrics
                'structure': structure,
                'model_params': {
                    'input_dim': input_dim, 'latent_factor': latent_factor,
                    'd_optimizer': default_opt(lr_d), 'g_optimizer': default_opt(lr_g)
                },
                'sampler': sampler,
                'training_params': {
                    'dataset': x, 'epochs': epochs, 'batch_size': batch_size,
                    'sample_interval': sample_interval, 'sampler': sampler,
                    'metrics': [JSD()]
                },
                'info': info
            }
            run_single_case(path, case, **kwargs)

        # write config done for this hparam
        info.done = True
        info.save(path / 'config.json')

    pass
