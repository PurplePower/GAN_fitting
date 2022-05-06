from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import time

from models.BaseGAN import BaseGAN
from visualizers.BaseSampler import BaseSampler
from utils.common import random_batch_getter
from utils.structures import level_1_structure


class SWG(BaseGAN):

    def __init__(
            self, input_dim, latent_factor, D=None, G=None,
            d_optimizer=None, g_optimizer=None,
            n_directions=100, use_discriminator=True, lambda1=1.0
    ):
        super().__init__(input_dim, latent_factor)
        self.use_discriminator = use_discriminator
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.n_directions = n_directions
        self.lambda1 = float(lambda1)  # gradient penalty coefficient

        super()._setup_models(D, G, d_optimizer, g_optimizer)

        if not self.use_discriminator:
            self.discriminator = self.d_optimizer = None
            raise Exception('SWG without discriminator can be severely bad in fitting anything.')
        else:
            # make discriminator a feature extractor
            self.discriminator = keras.Model(
                inputs=self.discriminator.input,
                outputs=[self.discriminator.layers[-2].output, self.discriminator.output])
            print(f'Discriminator has {self.discriminator.layers[-2].output_shape[1]} features')

    def _build_discriminator(self) -> keras.Sequential:
        return level_1_structure(self.input_dim, self.latent_factor)[0] if self.use_discriminator else None

    def _build_generator(self) -> keras.Sequential:
        return level_1_structure(self.input_dim, self.latent_factor)[1]

    def _build_d_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return Adam(1e-3) if self.use_discriminator else None

    def _build_g_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return Adam(1e-3)

    ###################################################
    #   losses and training
    ###################################################

    @tf.function
    def _sw_loss(self, true_features, fake_features):
        n_features = true_features.shape[1]

        # make random unit directions
        theta = tf.random.normal([n_features, self.n_directions])
        theta = tf.linalg.l2_normalize(theta, axis=0)  # unit direction vectors

        # use random orthogonal directions
        # a = tf.random.normal([n_features, n_features])
        # s, theta, v = tf.linalg.svd(a)

        # project features onto directions
        # shapes are [n_directions, batch_size]
        true_projected = tf.transpose(true_features @ theta)
        fake_projected = tf.transpose(fake_features @ theta)

        # sort along each dimension
        true_sorted = tf.sort(true_projected, axis=-1)
        fake_sorted = tf.sort(fake_projected, axis=-1)

        return tf.reduce_mean(tf.square(fake_sorted - true_sorted))

    @tf.function
    def _gradient_penalty(self, x_real, x_fake):
        t = tf.random.uniform(x_real.shape)
        interpolate = x_real * t + x_fake * (1.0 - t)

        with tf.GradientTape() as tape:
            tape.watch([interpolate])
            d_interpolate_logits = self.discriminator(interpolate)

        grads = tape.gradient(d_interpolate_logits, interpolate)
        return tf.reduce_mean(tf.square(tf.norm(grads, axis=1) - 1.0))

    @tf.function
    def _train_step(self, x_real):
        batch_size = x_real.shape[0]
        discriminator, generator = self.discriminator, self.generator
        disc_loss = .0

        with tf.GradientTape(persistent=True) as tape:
            z = tf.random.normal([batch_size, self.latent_factor])
            x_fake = generator(z, training=True)

            if self.use_discriminator:
                # use discriminator features for sw loss
                real_feat, real_prob = self.discriminator(x_real, training=True)
                fake_feat, fake_prob = self.discriminator(x_fake, training=True)

                # generator loss
                sw_loss = self._sw_loss(real_feat, fake_feat)

                # discriminator loss
                true_loss = self.bce(tf.ones_like(real_prob), real_prob)
                fake_loss = self.bce(tf.zeros_like(fake_prob), fake_prob)
                disc_loss = tf.reduce_mean(true_loss + fake_loss)
                gp = self._gradient_penalty(x_real, x_fake)
                disc_loss += self.lambda1 * gp

                pass

            else:
                # compute sliced Wasserstein loss
                sw_loss = self._sw_loss(x_real, x_fake)
                pass

        # update
        if self.use_discriminator:
            d_grads = tape.gradient(disc_loss, discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            pass

        g_grad = tape.gradient(sw_loss, generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))

        return disc_loss, sw_loss

    def train(
            self, dataset, epochs, batch_size=32,
            sample_interval=20, sampler: BaseSampler = None, sample_number=300,
            metrics=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        dataset = self._check_dataset(dataset)
        seed = tf.random.normal([sample_number, self.latent_factor])
        n_samples, n_batch = dataset.shape[0], dataset.shape[0] // batch_size
        metrics = metrics or []
        losses, metric_values = [], [[] for m in metrics]

        batch_getter = random_batch_getter(dataset, batch_size)

        for epoch in range(epochs):
            start = time.time()
            kwargs = {'model': self, 'dataset': dataset, 'epoch': epoch}

            total_d_loss = total_g_loss = .0

            for i in range(n_batch):
                d_loss, g_loss = self._train_step(next(batch_getter))
                total_d_loss += d_loss
                total_g_loss += g_loss
                pass

            if epoch % sample_interval == 0 and sampler is not None:
                sampler(self.generator(seed), epoch)
                for i, v in enumerate(metric_values):
                    v.append(metrics[i](**kwargs))

            losses.append((total_d_loss / n_batch, total_g_loss / n_batch))
            self.print_epoch(epoch, epochs, time.time() - start, *losses[-1])
            pass

        # last sample
        sampler(self.generator(seed), epochs - 1)
        self.trained_epoch += epochs

        return np.array(losses), np.array(metric_values)

    ######################################################
    #       Prediction and Generation
    ######################################################

    def estimate_prob(self, x, sample_points) -> np.ndarray:
        if not self.use_discriminator:
            return np.zeros_like(sample_points)
        features, probs = self.discriminator(sample_points, training=False)
        return np.array(probs)
