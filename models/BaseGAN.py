import abc
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Tuple

from visualizers.BaseSampler import BaseSampler
from utils import save


class BaseGAN(abc.ABC):

    def __init__(self, input_dim, latent_factor):
        assert input_dim > 0 and latent_factor > 0
        self.input_dim = input_dim
        self.latent_factor = latent_factor
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.trained_epoch = 0
        self.name = self.__class__.__name__
        pass

    def _setup_models(self, D=None, G=None, d_optimizer=None, g_optimizer=None):
        self.discriminator = D if D is not None else self._build_discriminator()
        self.generator = G if G is not None else self._build_generator()
        self.d_optimizer = d_optimizer if d_optimizer is not None else self._build_d_optimizer()
        self.g_optimizer = g_optimizer if g_optimizer is not None else self._build_g_optimizer()

    @abc.abstractmethod
    def _build_discriminator(self) -> keras.Sequential:
        pass

    @abc.abstractmethod
    def _build_generator(self) -> keras.Sequential:
        pass

    @abc.abstractmethod
    def _build_d_optimizer(self) -> tf.keras.optimizers.Optimizer:
        pass

    @abc.abstractmethod
    def _build_g_optimizer(self) -> tf.keras.optimizers.Optimizer:
        pass

    ###############################################
    #       Training
    ###############################################

    @abc.abstractmethod
    def train(self, dataset, epochs, batch_size=32, sample_interval=20,
              sampler: BaseSampler = None, sample_number=300, metrics=[]) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def _check_dataset(dataset):
        """
        Check dataset type and try converting to tf.Tensor.
        :param dataset:
        :return:
        """
        if isinstance(dataset, np.ndarray):
            return tf.constant(dataset)
        elif not isinstance(dataset, tf.Tensor):
            raise Exception(f'Currently not supported dataset as {type(dataset)}')

    def print_epoch(self, epoch, epochs, cost, d_loss, g_loss):
        print(f'Epoch {epoch + self.trained_epoch}/{epochs + self.trained_epoch} cost {cost:.2f} s, '
              f'D loss: {d_loss:.6f}, G loss: {g_loss:.6f}')

    ################################################
    #       Prediction and Generation
    ################################################

    def generate(self, n_samples=None, seed=None):
        """
        Generate points from learnt distribution.
        :param n_samples: Used when seed is None.
        :param seed: of shape (n_samples, latent_factor). If provided, generate by this seed.
        :return:
        """
        if seed is not None:
            seed_size = np.shape(seed)
            if seed_size[1] != self.latent_factor:
                raise Exception(
                    f'Incompatible latent factor size: expected {self.latent_factor}, got {seed_size[1]}')
            return self.generator(seed, training=False)
        elif n_samples is not None:
            seed = tf.random.normal([n_samples, self.latent_factor])
            return self.generator(seed, training=False)
        else:
            raise Exception('n_sample and seed can not be both None')

    ################################################
    #       Loading and Saving
    ################################################

    MAIN_MODEL_FILE = 'main_model'
    G_OPT_FILE = 'g_opt'
    D_OPT_FILE = 'd_opt'
    G_MODEL_DIR = 'generator'
    D_MODEL_DIR = 'discriminator'

    def save(self, path):
        if isinstance(path, str):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # make tf objects None and backup
        d_model, d_opt = self.discriminator, self.d_optimizer
        g_model, g_opt = self.generator, self.g_optimizer
        self.discriminator = self.generator = self.d_optimizer = self.g_optimizer = None

        # save the model
        with open(path / self.MAIN_MODEL_FILE, 'wb') as f:
            pickle.dump(self, f)

        # save tf objects
        save.save_optimizer(d_opt, path, self.D_OPT_FILE)
        save.save_optimizer(g_opt, path, self.G_OPT_FILE)

        if d_model:
            d_model.save(path / self.D_MODEL_DIR)
        if g_model:
            g_model.save(path / self.G_MODEL_DIR)

        # restore tf objects
        self.discriminator, self.d_optimizer = d_model, d_opt
        self.generator, self.g_optimizer = g_model, g_opt
        pass

    @classmethod
    def load(cls, path):
        if isinstance(path, str):
            path = Path(path)

        # load the model
        with open(path / cls.MAIN_MODEL_FILE, 'rb') as f:
            model = pickle.load(f)

        assert isinstance(model, BaseGAN) and model.name == cls.__name__  # assert name correct

        # load the tf objects
        model.discriminator = keras.models.load_model(path / cls.D_MODEL_DIR)
        model.generator = keras.models.load_model(path / cls.G_MODEL_DIR)
        model.d_optimizer = save.load_optimizer(path, cls.D_OPT_FILE, model=model.discriminator)
        model.g_optimizer = save.load_optimizer(path, cls.G_OPT_FILE, model=model.generator)

        return model
