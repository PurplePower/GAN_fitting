import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from typing import Union

from models.BaseGAN import BaseGAN
from visualizers.BaseSampler import BaseSampler
from utils.common import random_batch_getter


class GAN(BaseGAN):
    def __init__(self,
                 input_dim, latent_factor=5,
                 D=None, G=None, d_optimizer=None, g_optimizer=None):
        super().__init__(input_dim, latent_factor)
        super()._setup_models(D, G, d_optimizer, g_optimizer)

    def _build_discriminator(self):
        return keras.Sequential([
            layers.Dense(32, input_shape=(self.input_dim,)), layers.LeakyReLU(),
            # layers.Dense(32), layers.LeakyReLU(),
            layers.Dense(16), layers.LeakyReLU(),
            layers.Dense(1)
        ])

    def _build_generator(self):
        return keras.Sequential([
            layers.Dense(32, use_bias=False, input_shape=(self.latent_factor,)), layers.LeakyReLU(),
            # layers.Dense(32), layers.LeakyReLU(),
            layers.Dense(16), layers.LeakyReLU(),
            layers.Dense(self.input_dim)
        ])

    def _build_d_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.SGD(0.01)

    def _build_g_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.SGD(0.01)

    ###############################
    #   losses and training
    ###############################

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @staticmethod
    def _generator_loss(fake_output):
        return GAN.cross_entropy(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def _discriminator_loss(real_output, fake_output):
        real_loss = GAN.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = GAN.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @tf.function
    def _train_step_discriminator(self, real_x):
        print('Tracing d_step...')
        discriminator, generator = self.discriminator, self.generator
        noise = tf.random.normal([len(real_x), self.latent_factor])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_x = generator(noise, training=True)  # training=True for differentiable D
            real_output = discriminator(real_x, training=True)
            fake_output = discriminator(fake_x, training=True)

            disc_loss = self._discriminator_loss(real_output, fake_output)

        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def _train_step_both(self, real_x):
        print('Tracing both_step...')  # tf.function trace for only a few times
        discriminator, generator = self.discriminator, self.generator
        noise = tf.random.normal([len(real_x), self.latent_factor])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_x = generator(noise, training=True)  # training=True for differentiable D
            real_output = discriminator(real_x, training=True)
            fake_output = discriminator(fake_x, training=True)

            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        # update gradient
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        return disc_loss, gen_loss

    def train(self, dataset: Union[tf.Tensor, np.ndarray], epochs, batch_size=32, sample_interval=20,
              sampler: BaseSampler = None, sample_number=300, metrics=[],
              dg_train_ratio=1):
        dataset = self._check_dataset(dataset)
        seed = tf.random.normal([sample_number, self.latent_factor])
        n_samples = dataset.shape[0]
        n_batch = n_samples // batch_size
        losses, metric_values = [], [[] for m in metrics]

        batch_getter = random_batch_getter(dataset, batch_size)

        for epoch in range(epochs):
            start = time.time()
            kwargs = {'generator': self.generator, 'discriminator': self.discriminator,
                      'model': self, 'dataset': dataset}
            total_d_loss = total_g_loss = .0

            # in each batch, train D for dg_train_ratio times and G once
            with tf.profiler.experimental.Trace('train', step_num=epoch, _r=1):
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

            total_g_loss /= n_batch
            total_d_loss /= n_batch
            losses.append((total_d_loss, total_g_loss))
            self.print_epoch(epoch, epochs, time.time() - start, total_d_loss, total_g_loss)

        self.trained_epoch += epochs

        return np.array(losses), np.array(metric_values)

    def _train_deprecated(self, dataset, epochs, batch_size=32, sample_interval=20, sampler: BaseSampler = None,
                          sample_number=300,
                          dg_train_ratio=1):
        """
        Deprecated. dataset with tf.data.Dataset without tf.function is slow when dg_train_ratio > 1
        """
        seed = tf.random.normal([sample_number, self.latent_factor])
        dataset = dataset.shuffle(len(dataset)).repeat(dg_train_ratio).batch(batch_size, drop_remainder=True)
        n_batch = len(dataset)
        losses = []  # save tuple (d_loss, g_loss) of each epoch

        for epoch in range(epochs):
            start = time.time()

            with tf.profiler.experimental.Trace('train', step_num=epoch, _r=1):
                total_g_loss = total_d_loss = 0.0
                i = dg_train_ratio - 1
                for through_dataset in range(dg_train_ratio):
                    for batch in dataset:
                        if i == 0:
                            # s = time.time()
                            d_loss, g_loss = self._train_step_both(batch)
                            # print(f'train step_d cost {time.time() - s:.3f} s')
                            total_d_loss += d_loss
                            total_g_loss += g_loss
                            i = dg_train_ratio - 1  # reset counter
                        else:
                            self._train_step_discriminator(batch)
                            i -= 1

            if epoch % sample_interval == 0 and sampler is not None:
                sampler(self.generator(seed), epoch)

            total_g_loss /= n_batch
            total_d_loss /= n_batch
            losses.append((total_d_loss, total_g_loss))
            self.print_epoch(epoch, epochs, time.time() - start, total_d_loss, total_g_loss)

        self.trained_epoch += epochs

        return np.array(losses)
