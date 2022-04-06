import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from typing import Union

from models.BaseGAN import BaseGAN
from visualizers.BaseSampler import BaseSampler
from utils.common import random_batch_getter


class WGAN(BaseGAN):
    def __init__(self, input_dim, latent_factor=5, D=None, G=None, d_optimizer=None, g_optimizer=None):
        super().__init__(input_dim, latent_factor)
        super()._setup_models(D, G, d_optimizer, g_optimizer)

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

    ##################################
    #   losses and training
    ##################################

    @tf.function
    def _gradient_penalty(self, x_real, x_fake):
        batchsz = x_real.shape[0]

        t = tf.random.uniform([batchsz, 1])
        interpolate = t * x_real
        interpolate += (1 - t) * x_fake

        with tf.GradientTape() as tape:
            tape.watch([interpolate])
            d_interpolate_logits = self.discriminator(interpolate)

        grads = tape.gradient(d_interpolate_logits, interpolate)  # same shape as interpolate, (batchsz, input_dim)
        gp = tf.norm(grads, axis=1)
        gp = tf.reduce_mean((gp - 1) ** 2)
        return gp

    @tf.function
    def _train_step_both(self, x_real):
        noise = tf.random.normal([len(x_real), self.latent_factor])
        discriminator, generator = self.discriminator, self.generator

        with tf.GradientTape(persistent=True) as tape:
            x_fake = generator(noise, training=True)

            gp = self._gradient_penalty(x_real, x_fake)

            # TODO: combine as one
            d_fake_logits = discriminator(x_fake, training=True)
            d_real_logits = discriminator(x_real, training=True)

            d_loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 1.0 * gp
            g_loss = - tf.reduce_mean(d_fake_logits)

        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        return d_loss, g_loss

    @tf.function
    def _train_step_discriminator(self, x_real):
        noise = tf.random.normal([len(x_real), self.latent_factor])
        discriminator, generator = self.discriminator, self.generator

        with tf.GradientTape(persistent=True) as tape:
            x_fake = generator(noise, training=True)
            gp = self._gradient_penalty(x_real, x_fake)
            d_fake_logits = discriminator(x_fake, training=True)
            d_real_logits = discriminator(x_real, training=True)
            d_loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 1.0 * gp

        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        return d_loss

    def train(self, dataset: Union[tf.Tensor, np.array], epochs, batch_size=32, sample_interval=20,
              sampler: BaseSampler = None, sample_number=300, dg_train_ratio=1, metrics=[]):
        dataset = self._check_dataset(dataset)
        seed = tf.random.normal([sample_number, self.latent_factor])
        n_samples, n_batch = dataset.shape[0], dataset.shape[0] // batch_size
        losses, metric_values = [], [[] for m in metrics]

        batch_getter = random_batch_getter(dataset, batch_size)

        for epoch in range(epochs):
            start = time.time()
            kwargs = {'generator': self.generator, 'discriminator': self.discriminator,
                      'model': self, 'dataset': dataset}
            total_g_loss = total_d_loss = 0.0

            for i in range(n_batch):
                for _ in range(dg_train_ratio - 1):
                    self._train_step_discriminator(next(batch_getter))
                    pass

                d_loss, g_loss = self._train_step_both(next(batch_getter))
                total_d_loss += d_loss
                total_g_loss += g_loss

            if epoch % sample_interval == 0 and sampler is not None:
                sampler(self.generator(seed), epoch)
                for i, v in enumerate(metric_values):
                    v.append(metrics[i](**kwargs))

            losses.append((total_d_loss / n_batch, total_g_loss / n_batch))
            self.print_epoch(epoch, epochs, time.time() - start, total_d_loss / n_batch, total_g_loss / n_batch)

        self.trained_epoch += epochs

        return np.array(losses), np.array(metric_values)
