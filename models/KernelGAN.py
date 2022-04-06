import time
from typing import Tuple
from pathlib import Path
import pickle
import json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.layers import Dense, LeakyReLU
from tensorflow import math

from models import BaseGAN
from visualizers.BaseSampler import BaseSampler
from utils.common import random_batch_getter
from utils import save


class KernelGAN(BaseGAN):
    def __init__(self, input_dim, latent_factor=5,
                 D=None, G=None, d_optimizer=None, g_optimizer=None,
                 bandwidth=0.5, bandwidth_updater=None):
        super().__init__(input_dim, latent_factor)
        super()._setup_models(D, G, d_optimizer, g_optimizer)
        self.mvn = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros([input_dim]), scale_diag=tf.ones([input_dim]))
        self.bandwidth = bandwidth
        self.bandwidth_updater = bandwidth_updater
        self.regularizer = 0
        self._kernel_constant = 1 / tf.sqrt((2 * np.pi) ** latent_factor)

    def _build_discriminator(self) -> keras.Sequential:
        return None

    def _build_generator(self) -> keras.Sequential:
        return keras.Sequential([
            Dense(32, input_shape=(self.latent_factor,)), LeakyReLU(),
            Dense(16), LeakyReLU(),
            Dense(self.input_dim)
        ])

    def _build_d_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return None

    def _build_g_optimizer(self) -> tf.keras.optimizers.Optimizer:
        # TODO: SGD or Adam
        return keras.optimizers.SGD(1e-3)

    ###############################
    #    kernel estimations
    ###############################

    @tf.function
    def _kernel(self, x):
        """
        Multivariate Gaussian kernel
        :param x: samples of points passed to kernel
        :return: PDF of points
        """
        t1 = self._kernel_constant * tf.exp(-0.5 * (tf.norm(x, axis=1) ** 2))
        # t2 = self.mvn.prob(x)
        return t1

    @tf.function
    def _estimate_prob(self, x, x_sample):
        """
        Compute estimated p_{n, bandwidth} or p^{(\theta)}_{n, bandwidth}
        :param x: point(s) where density is estimated
        :param x_sample: samples of points drawn from real/fake distribution
        :return:
        """
        return tf.reduce_mean(self._kernel((x_sample - x) / self.bandwidth))

    @tf.function
    def _kernel_loss(self, x_real, x_fake):
        """
        [Deprecated] Compute kernel loss. Low performance due to little parallelization.
        :param x_real:
        :param x_fake:
        :return:
        """
        phi = self.regularizer
        n = x_real.shape[0]
        estimate_prob = self._estimate_prob

        loss = 0.0
        for i in range(n):
            # discriminator loss
            p_real_xi = estimate_prob(x_real[i], x_real)  # p_real(Xi)
            p_fake_xi = estimate_prob(x_real[i], x_fake)  # p_fake^{(\theta)}(Xi)

            # generator loss
            p_fake_xg = estimate_prob(x_fake[i], x_fake)  # p_fake(g(Zi))
            p_real_xg = estimate_prob(x_fake[i], x_real)  # p_real(g(Zi))

            loss += math.log((p_real_xi + phi) / (p_real_xi + p_fake_xi + 2 * phi)) + \
                    math.log((p_fake_xg + phi) / (p_real_xg + p_fake_xg + 2 * phi))
            pass
        loss = loss / n
        return loss

    @tf.function
    def _kernel_loss2(self, x_real, x_fake):
        """
        Compute kernel loss, kernel gan's objective.
        :param x_real:
        :param x_fake:
        :return:
        """
        phi = self.regularizer
        n = x_real.shape[0]
        estimate_prob = self._estimate_prob2

        # making cartesian products
        tiles = tf.constant([n, 1])
        x_real_tiled = tf.tile(x_real, tiles)
        x_fake_tiled = tf.tile(x_fake, tiles)
        x_real_repeated = tf.repeat(x_real, n, axis=0)
        x_fake_repeated = tf.repeat(x_fake, n, axis=0)

        p_real_xi = estimate_prob(x_real_repeated, x_real_tiled, n)
        p_fake_xi = estimate_prob(x_real_repeated, x_fake_tiled, n)
        p_real_xg = estimate_prob(x_fake_repeated, x_real_tiled, n)
        p_fake_xg = estimate_prob(x_fake_repeated, x_fake_tiled, n)

        d_loss = math.log((p_real_xi + phi) / (p_real_xi + p_fake_xi + 2 * phi))
        g_loss = math.log((p_fake_xg + phi) / (p_real_xg + p_fake_xg + 2 * phi))
        return tf.reduce_mean(d_loss + g_loss)

    @tf.function
    def _estimate_prob2(self, repeated, tiled, n):
        t = self._kernel((repeated - tiled) / self.bandwidth)  # of shape (n*n, 1)
        t = tf.reshape(t, [n, n])
        return tf.reduce_mean(t, axis=1)

    ###############################
    #    training
    ###############################

    @tf.function
    def _train_step(self, x_real):
        n = x_real.shape[0]
        noise = tf.random.normal([n, self.latent_factor])
        generator = self.generator

        with tf.GradientTape() as tape:
            x_fake = generator(noise, training=True)
            loss = self._kernel_loss2(x_real, x_fake)

        # only generator is updated
        grads = tape.gradient(loss, generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        return loss

    def train(self, dataset, epochs, batch_size=32, sample_interval=20, sampler: BaseSampler = None, sample_number=300,
              metrics=[]) -> Tuple[np.ndarray, np.ndarray]:
        dataset = self._check_dataset(dataset)
        seed = tf.random.normal([sample_number, self.latent_factor])
        n_samples, n_batch = dataset.shape[0], dataset.shape[0] // batch_size
        losses, metric_values = [], [[] for m in metrics]
        batch_getter = random_batch_getter(dataset, batch_size)

        bw_backup = self.bandwidth

        for epoch in range(epochs):
            start = time.time()
            kwargs = {'generator': self.generator, 'discriminator': self.discriminator,
                      'model': self, 'dataset': dataset}
            total_g_loss = total_d_loss = 0.0

            for i in range(n_batch):
                g_loss = self._train_step(next(batch_getter))
                total_g_loss += g_loss

            if epoch % sample_interval == 0 and sampler is not None:
                sampler(self.generator(seed), epoch)
                for i, v in enumerate(metric_values):
                    v.append(metrics[i](**kwargs))

            # update bandwidth
            if self.bandwidth_updater:
                self.bandwidth = self.bandwidth_updater(self.bandwidth, epoch, epochs)

            total_g_loss /= n_batch
            losses.append((total_d_loss, total_g_loss))
            self.print_epoch(epoch, epochs, time.time() - start, total_d_loss, total_g_loss)

        self.bandwidth = bw_backup
        self.trained_epoch += epochs
        return np.array(losses), np.array(metric_values)

    @classmethod
    def load(cls, path):
        if isinstance(path, str):
            path = Path(path)

        # load the model
        with open(path / cls.MAIN_MODEL_FILE, 'rb') as f:
            model = pickle.load(f)

        assert isinstance(model, BaseGAN) and model.name == cls.__name__  # assert name correct

        # load the tf objects
        model.discriminator = None
        model.generator = keras.models.load_model(path / cls.G_MODEL_DIR)
        model.d_optimizer = None
        model.g_optimizer = save.load_optimizer(path, cls.G_OPT_FILE, model=model.generator)

        return model

    ###############################
    #    discriminator estimate
    ###############################

    def estimate_prob(self, x_real, points):
        """
        Compute p_hat_{n, /sigma}(x), i.e, points' kernel density in x_real.
        For plot 2d discriminator judge area.
        :param x_real:
        :param points:
        :return:
        """

        n = x_real.shape[0]
        points = tf.cast(points, x_real.dtype)  # agree on this type

        # make cartesian product of points * x_real
        points_repeated = tf.repeat(points, n, axis=0)
        x_real_tiled = tf.tile(x_real, tf.constant([points.shape[0], 1]))

        p_estimate = self._kernel((points_repeated - x_real_tiled) / self.bandwidth)  # [n1 * n2, 1]
        p_estimate = tf.reshape(p_estimate, [points.shape[0], n])
        p_estimate = tf.reduce_mean(p_estimate, axis=1)

        return p_estimate
