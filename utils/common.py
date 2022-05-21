import os
import glob
from pathlib import Path
import shutil
import tensorflow as tf
import numpy as np
from functools import partial


def clean_images(path):
    files = glob.glob(path + '/*.png')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def random_batch_getter(dataset: tf.Tensor, batch_size):
    """
    Get infinite batches from `dataset` of size `batch_size`, remainders
    are dropped.
    """
    ds = tf.random.shuffle(dataset)
    n_batch = dataset.shape[0] // batch_size  # this drops remainders
    while True:
        for batch_no in range(n_batch):
            yield ds[(t := batch_no * batch_size):t + batch_size]
        ds = tf.random.shuffle(ds)
    pass


def empty_directory(path):
    if not isinstance(path, Path):
        path = Path(path)

    for f in path.glob('*'):
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f)
    return


def round_to_fit(a):
    """
    Round float `a` to correct string. Value 0.001 may be printed
    as 0.00100000...00474..., which is not desirable.
    :param a:
    :return:
    """
    return np.format_float_positional(a)  # unique=True


def get_optimizer_string(opt, scientific=False):
    """
    Convert optimizer to readable string.
    :param opt:
    :param scientific:
    :return:
    """
    if scientific:
        f = partial(np.format_float_scientific, trim='-', exp_digits=1)

        def formatter(x):
            return x if x == 1 else formatter(x)
    else:
        formatter = partial(np.format_float_positional, trim='-')

    name = opt.__class__.__name__

    lr = opt.learning_rate
    if isinstance(lr, tf.Variable):
        scheduler = formatter(lr.numpy())
    else:
        # a schedule
        import tensorflow.keras.optimizers.schedules as schedules
        if isinstance(
                lr,
                (schedules.ExponentialDecay, schedules.InverseTimeDecay)):
            scheduler = f'{lr.__class__.__name__}(' \
                        f'{formatter(lr.initial_learning_rate)},' \
                        f'{lr.decay_steps},' \
                        f'{formatter(lr.decay_rate)})'
            # may add staircase

        elif isinstance(lr, schedules.PiecewiseConstantDecay):
            scheduler = f'{lr.__class__.__name__}(' \
                        f'{lr.boundaries},{lr.values})'
        elif isinstance(lr, schedules.PolynomialDecay):
            scheduler = f'{lr.__class__.__name__}(' \
                        f'{lr.decay_steps})'
        elif isinstance(lr, schedules.PiecewiseConstantDecay):
            s = str(list(map(formatter, lr.values))).replace("'", "")
            s = s.replace('"', '')
            scheduler = f'{lr.__class__.__name__}(' \
                        f'{lr.boundaries},' \
                        f'{s})'

        else:
            scheduler = 'sch'

    return f'{name}({scheduler})'
