import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
from pathlib import Path

from models import WGAN, BaseGAN
from data.datamaker import *
from metrics.JensenShannonDivergence import JensenShannonDivergence as JSD
from utils.common import empty_directory
from utils.structures import *
from utils.runtests import *


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
    model = WGAN(D=D, G=G, **model_params)

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
    parent_path = Path('htest')

    # model hyper-params
    input_dim = 2
    default_opt = keras.optimizers.SGD
    structures = [level_1_structure, level_2_structure]

    # training params
    learning_rate_combinations = [(1e-2, 1e-2), (1e-3, 1e-3)]
    latent_factors = [5]
    dg_train_ratios = [1, 3]
    epochs = 10000
    batch_size = 64

    #
    repeat_times = 3  # cases per hparam
    sample_interval = 20

    x, sampler = make_ring_dots(1024, path='')  # one dataset for all trainings

    for (lr_d, lr_g), latent_factor, structure, dg_r in itertools.product(learning_rate_combinations, latent_factors,
                                                                          structures, dg_train_ratios):
        D, G = structure(input_dim, latent_factor)
        model = WGAN(input_dim, latent_factor, D=D, G=G,
                    d_optimizer=default_opt(lr_d), g_optimizer=default_opt(lr_g))

        # pack training and model information
        info = Info(
            gan_type='WGAN',
            input_dim=input_dim, latent_factor=latent_factor, opt=default_opt.__name__,
            lr_d=lr_d, lr_g=lr_g, batch_sz=batch_size, dg_r=dg_r, struct=STRUCTURE_NAMES[structure],
            trained_epochs=epochs
        )

        folder_name = folder_name_getter(info)
        path = parent_path / folder_name
        cases_to_run = get_cases_to_run(path, repeat_times)

        print(f'Training cases {cases_to_run} will run in path {path}')
        if not cases_to_run:
            continue

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
