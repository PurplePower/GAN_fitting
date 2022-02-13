import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
from pathlib import Path


def save_optimizer(opt: keras.optimizers.Optimizer, path, filename):
    data = {
        'config': opt.get_config(),
        'weights': opt.get_weights()
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

    opt = keras.optimizers.get(data['config']['name']).from_config(data['config'])

    # apply zero-grad to shape its weights correctly
    shape_weights = [tf.Variable(tf.zeros_like(w)) for w in model.trainable_variables]
    zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
    opt.apply_gradients(zip(zero_grads, shape_weights))

    opt.set_weights(data['weights'])
    return opt
