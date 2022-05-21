from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import time

from models.BaseGAN import BaseGAN
from visualizers.BaseSampler import BaseSampler
from utils.common import random_batch_getter
from utils.structures import level_1_structure


class SWGAN(BaseGAN):
    """
    Implements SWGAN described in paper Sliced Wasserstein Generative Models, Jiqing Wu et al.

    Using orthogonal projection and update in Stiefel manifold.

    [Currently Deprecated] unstable training.

    """

    def __init__(
            self, input_dim, latent_factor, D=None, G=None,
            d_optimizer=None, g_optimizer=None,
            lambda1=1.0, lambda2=1.0
    ):
        super().__init__(input_dim, latent_factor)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

        super()._setup_models(D, G, d_optimizer, g_optimizer)
        self._construct_discriminator()

        self._init_params()

    def _construct_discriminator(self):
        self.discriminator = keras.Model(
            inputs=self.discriminator.input,
            outputs=self.discriminator.layers[-2].output
        )  # encoder only
        self.feat_dim = self.discriminator.output_shape[1]
        print(f'Discriminator has {self.feat_dim} features')

    def _init_params(self):
        self._swd_name_pfx = 'SWD_block'
        self.swd_blocks = [
            tf.Variable(
                np.linalg.svd(np.random.normal(0, 1, (self.feat_dim, self.feat_dim)))[0],
                dtype=tf.float32,
                name=f'{self._swd_name_pfx}{i}'
            )
            for i in range(4)
        ]
        self.swd_leaky = LeakyReLU()
        pass

    def _build_discriminator(self) -> keras.Sequential:
        return level_1_structure(self.input_dim, self.latent_factor)[0]

    def _build_generator(self) -> keras.Sequential:
        return level_1_structure(self.input_dim, self.latent_factor)[1]

    def _build_d_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return Adam(1e-5)

    def _build_g_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return Adam(1e-5)

    #####################################################
    #       losses and training
    #####################################################

    @tf.function
    def _train_step(self, x_real):
        batch_size = x_real.shape[0]
        discriminator, generator = self.discriminator, self.generator

        with tf.GradientTape(persistent=True) as tape:
            z = tf.random.normal([batch_size, self.latent_factor])
            x_fake = generator(z, training=True)

            # encode to features
            real_feat = discriminator(x_real, training=True)
            fake_feat = discriminator(x_fake, training=True)

            # SWD blocks
            disc_real = tf.reduce_mean(self._swd_blocks_compute(real_feat), axis=1)
            disc_fake = tf.reduce_mean(self._swd_blocks_compute(fake_feat), axis=1)

            # losses
            disc_loss = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
            gen_loss = tf.reduce_mean(disc_fake)

            # GP1
            disc_loss += self._gradient_penalty_1(x_real, x_fake) * self.lambda1

            # GP2
            # disc_loss += self._gradient_penalty_2(real_feat, fake_feat) * self.lambda2

        # generator update as usual
        gen_grad = tape.gradient(gen_loss, generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))

        # SWD blocks gradients on Stiefel manifold
        disc_grad = tape.gradient(disc_loss, discriminator.trainable_variables)

        # tangential components
        swd_grad = tape.gradient(disc_loss, self.swd_blocks)
        for i, (grad, var) in enumerate(zip(swd_grad, self.swd_blocks)):
            tmp1 = tf.transpose(var) @ grad
            tmp2 = (tmp1 + tf.transpose(tmp1)) / 2
            tmp_grad = grad - tf.matmul(var, tmp2)
            swd_grad[i] = tmp_grad

        grads = [*disc_grad, *swd_grad]
        self.d_optimizer.apply_gradients(zip(grads, [*discriminator.trainable_variables, *self.swd_blocks]))

        # stiefel update
        for block in self.swd_blocks:
            q, _ = tf.linalg.qr(block)
            block.assign(q)
            pass

        return disc_loss, gen_loss

    @tf.function
    def _swd_blocks_compute(self, feat):
        # project
        proj = [feat @ w for w in self.swd_blocks]

        # concatenate to [batch_size, total_projections]
        proj = tf.concat(proj, 1)

        # element-wise LeakyReLU
        return self.swd_leaky(proj)

    @tf.function
    def _gradient_penalty_1(self, x_real, x_fake):
        with tf.GradientTape() as tape:
            t = tf.random.uniform([x_real.shape[0], 1])
            interpolated = x_real * t + x_fake * (1.0 - t)
            tape.watch([interpolated])
            loss = tf.reduce_mean(
                self._swd_blocks_compute(self.discriminator(interpolated)), axis=1)

        grads = tape.gradient(loss, interpolated)
        norm = tf.reduce_sum(tf.square(grads), axis=1)
        return tf.reduce_mean(norm)

    @tf.function
    def _gradient_penalty_2(self, real_feat, fake_feat):
        t = tf.random.uniform([real_feat.shape[0], 1])

        with tf.GradientTape() as tape:
            interpolated = real_feat * t + fake_feat * (1 - t)
            tape.watch([interpolated])
            activated = self._swd_blocks_compute(interpolated)

        grad = tape.gradient(activated, interpolated)
        return tf.reduce_mean(tf.square(grad - 0.001))



    def train(
            self, dataset, epochs, batch_size=64,
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
                # for _ in range(dg_train_ratio - 1):
                #     self._train_step_discriminator(next(batch_getter))
                #     pass

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
