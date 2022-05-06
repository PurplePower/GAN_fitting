import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from pathlib import Path


def save_optimizer(opt: keras.optimizers.Optimizer, path, filename):
    data = {
        'config': opt.get_config() if opt else None,
        'weights': opt.get_weights() if opt else None
    }
    if not isinstance(path, Path):
        path = Path(filename)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / filename, 'wb') as f:
        pickle.dump(data, f)
    pass


def load_optimizer(path, filename, model) -> keras.optimizers.Optimizer:
    if not isinstance(path, Path):
        path = Path(path)
    with open(path / filename, 'rb') as f:
        data = pickle.load(f)

    if not data['config']:
        return None

    opt = keras.optimizers.get(data['config']['name']).from_config(data['config'])

    # apply zero-grad to shape its weights correctly
    shape_weights = [tf.Variable(tf.zeros_like(w)) for w in model.trainable_variables]
    zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
    opt.apply_gradients(zip(zero_grads, shape_weights))

    opt.set_weights(data['weights'])
    return opt


def convert_numpy_types_to_natives(d: dict, copy_dict=True) -> dict:
    """
    If any of the value in dict (recursively) is of numpy types, which is
    not json serializable, then convert it to native int or float.
    The modification can be select in-place or on a new copy.
    :param d:
    :param copy_dict:
    :return:
    """
    if copy_dict:
        d = d.copy()

    numpy_ints = (np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)
    numpy_floats = (np.float16, np.float32, np.float64)

    for k, v in d.items():
        if isinstance(v, numpy_ints):
            d[k] = int(v)
        elif isinstance(v, numpy_floats):
            d[k] = float(v)
        elif isinstance(v, dict):
            d[k] = convert_numpy_types_to_natives(v)

    return d
