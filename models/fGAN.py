import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Union
import time

from models import BaseGAN
from visualizers.BaseSampler import BaseSampler
from utils.common import random_batch_getter


#########################################
# KL
#########################################

@tf.function
def kl_activate(v):
    return v


@tf.function
def kl_conjugate_activate(v):
    return tf.exp(v - 1.0)


#########################################
# Jensen-Shannon
#########################################

@tf.function
def jensen_shannon_activation(v):
    """
    :param v: A vector of probabilities output by D, each represents probability that this sample
                comes from real distribution.
    :return:
    """
    return tf.math.log(2.0) - tf.math.log(1.0 + tf.exp(-v))


@tf.function
def jensen_shannon_conjugate_activate(v):
    """
    :param v: Output of D
    :return:
    """
    # return - tf.math.log(2.0 - tf.exp(t))

    # f*(g_f(v)) = v + ln(1+exp(-v)) - ln2, make log = ln
    return v + tf.math.log(1 + tf.exp(-v)) - tf.math.log(2.0)


#########################################
# Pearson Chi-Square
#########################################

@tf.function
def pearson_chi_square_activate(v):
    return v


@tf.function
def pearson_chi_square_conjugate_activate(t):
    return tf.square(t) / 4.0 + t


#########################################
# Squared Hellinger
#########################################

@tf.function
def squared_hellinger_activate(v):
    return 1.0 - tf.exp(-v)


@tf.function
def squared_hellinger_conjugate_activate(v):
    return tf.exp(v) - 1.0
    # return t / (1.0 - t)


VARIATIONAL_DIVERGENCE = {
    'jensen-shannon': (jensen_shannon_activation, jensen_shannon_conjugate_activate),
    'pearson-chi-square': (pearson_chi_square_activate, pearson_chi_square_conjugate_activate),
    'squared-hellinger': (squared_hellinger_activate, squared_hellinger_conjugate_activate),
    'kl': (kl_activate, kl_conjugate_activate)

}


class fGAN(BaseGAN):
    def __init__(self, input_dim, latent_factor=5,
                 D=None, G=None, d_optimizer=None, g_optimizer=None, divergence: str = 'jensen-shannon'):
        super().__init__(input_dim, latent_factor)
        super()._setup_models(D, G, d_optimizer, g_optimizer)

        if divergence.lower() not in VARIATIONAL_DIVERGENCE:
            raise Exception(f'No such divergence or not implemented. '
                            f'Supports: {list(VARIATIONAL_DIVERGENCE.keys())}')

        self.divergence = divergence.lower()
        self.activation, self.conjugate_activate = VARIATIONAL_DIVERGENCE[self.divergence]
        # self.disc_grad_record = self.gen_grad_record = None

    def _build_discriminator(self) -> keras.Sequential:
        return keras.Sequential([
            layers.Dense(32, input_shape=(self.input_dim,)), layers.LeakyReLU(),
            layers.Dense(16), layers.LeakyReLU(),
            layers.Dense(1)
        ])

    def _build_generator(self) -> keras.Sequential:
        return keras.Sequential([
            layers.Dense(32, use_bias=False, input_shape=(self.latent_factor,)), layers.LeakyReLU(),
            layers.Dense(16), layers.LeakyReLU(),
            layers.Dense(self.input_dim)
        ])

    def _build_d_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.SGD(0.01)

    def _build_g_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.SGD(0.01)

    ###############################
    #

    # @tf.function
    # def _f_loss(self, x_real, x_fake):
    #     """
    #     Compute F(\theta, \omega).
    #     :param x_real:
    #     :param x_fake:
    #     :return:
    #     """
    #
    #     # term that records error from discriminating x_real
    #     real_loss = tf.reduce_mean(
    #         self.activation(self.discriminator(x_real, training=True)))
    #
    #     # term that records error from discriminating x_fake
    #     fake_loss = -1.0 * tf.reduce_mean(
    #         self.conjugate_f(self.activation(self.discriminator(x_fake, training=True))))
    #
    #     return real_loss + fake_loss

    @tf.function
    def _train_step(self, x_real, record_grads=False):
        batch_sz = len(x_real)
        noise = tf.random.normal([batch_sz, self.latent_factor])
        # discriminator, generator = self.discriminator, self.generator

        with tf.GradientTape(persistent=True) as tape:
            x_fake = self.generator(noise, training=True)
            # term that records error from discriminating x_real
            real_loss = tf.reduce_mean(
                self.activation(self.discriminator(x_real, training=True)))

            # term that records error from discriminating x_fake
            fake_loss = -1.0 * tf.reduce_mean(
                self.conjugate_activate(self.discriminator(x_fake, training=True)))

            objective = real_loss + fake_loss
            minus_objective = -objective

        disc_grads = tape.gradient(minus_objective, self.discriminator.trainable_variables)  # maximize
        gen_grads = tape.gradient(objective, self.generator.trainable_variables)  # minimize

        self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        if record_grads:
            d_norm = [tf.norm(g, ord=2) for g in disc_grads]
            g_norm = [tf.norm(g, ord=2) for g in gen_grads]
            return objective, d_norm, g_norm

        return objective

    def train(
            self, dataset, epochs, batch_size=32, sample_interval=20, sampler: BaseSampler = None, sample_number=300,
            metrics=[], record_grads=False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, list]]:
        dataset = self._check_dataset(dataset)
        seed = tf.random.normal([sample_number, self.latent_factor])
        n_samples = dataset.shape[0]
        n_batch = n_samples // batch_size
        losses, metric_values = [], [[] for m in metrics]
        grad_norms = []

        batch_getter = random_batch_getter(dataset, batch_size)

        for epoch in range(epochs):
            start = time.time()
            kwargs = {'generator': self.generator, 'discriminator': self.discriminator,
                      'model': self, 'dataset': dataset}
            total_objective = 0.0
            tmp_d_norm, tmp_g_norm = [], []

            for i in range(n_batch):
                results = self._train_step(next(batch_getter), record_grads=record_grads)
                objective = results[0] if record_grads else results
                total_objective += objective
                if record_grads:
                    tmp_d_norm.append(results[1])
                    tmp_g_norm.append(results[2])
                pass

            if record_grads:
                grad_norms.append((tf.reduce_mean(tmp_d_norm, axis=1), tf.reduce_mean(tmp_g_norm, axis=1)))

            if epoch % sample_interval == 0 and sampler is not None:
                sampler(self.generator(seed), epoch)
                for i, v in enumerate(metric_values):
                    v.append(metrics[i](**kwargs))

            losses.append((total_objective / batch_size, 0))
            self.print_epoch(epoch, epochs, time.time() - start, total_objective / batch_size, 0)

        self.trained_epoch += epochs

        if record_grads:
            return np.array(losses), np.array(metric_values), grad_norms
        else:
            return np.array(losses), np.array(metric_values)

    def save(self, path):
        self.activation = self.conjugate_activate = None
        super().save(path)
        self.activation, self.conjugate_activate = VARIATIONAL_DIVERGENCE[self.divergence]

    @classmethod
    def load(cls, path):
        model: fGAN = super().load(path)
        model.activation, model.conjugate_activate = VARIATIONAL_DIVERGENCE[model.divergence]
        return model
