import time

import tensorflow as tf
import numpy as np
from typing import Union, Tuple
from models import GAN
from visualizers.BaseSampler import BaseSampler


class GAN_Tensor(GAN):

    def _make_tf_dataset(self, x):
        # TODO: check type(x)
        return tf.data.Dataset.from_tensor_slices(x)

    def train(self, dataset: Union[tf.Tensor, np.ndarray], epochs, batch_size=64, sample_interval=20,
              sampler: BaseSampler = None, sample_number=300, metrics=None,
              dg_train_ratio=1):
        n_samples = dataset.shape[0]
        # dataset = self._check_dataset(dataset)
        dataset = self._make_tf_dataset(dataset)
        seed = tf.random.normal([sample_number, self.latent_factor])
        n_batch = int(np.ceil(n_samples / batch_size))

        return_every_epochs = 2 * sample_interval

        metrics = metrics or []
        losses = []
        epoch = 0

        dataset = dataset.repeat(dg_train_ratio).batch(batch_size, drop_remainder=True)

        for i in range(epochs // return_every_epochs):
            start = time.time()
            local_loss = self.train_faster(
                dataset, epochs, sample_interval, dg_train_ratio,
                seed, n_batch, batch_size, epoch, return_every_epochs
            )
            local_loss = local_loss.numpy()
            # generated = generated.numpy()

            # record losses
            losses.extend(local_loss)

            # sampling
            # for j, samples in enumerate(generated):
            #     sampler(samples, epoch + j * sample_interval)

            avg_time = (time.time() - start) / return_every_epochs
            for j, loss in enumerate(local_loss):
                self.print_epoch(epoch + j, epochs, avg_time, loss[0], loss[1])

            epoch += return_every_epochs
            pass

        # TODO: last sample
        return losses, []

    # use tf.range() instead of python built-in range()
    @tf.function
    def train_faster(
            self, dataset: tf.data.Dataset, epochs, sample_interval,
            dg_train_ratio,
            seed, n_batch, batch_size, cur_epoch, return_every_epochs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        [Deprecated] no much improvement.

        :param dataset:
        :param epochs:
        :param sample_interval:
        :param dg_train_ratio:
        :param seed:
        :param n_batch:
        :param cur_epoch:
        :param return_every_epochs:
        :return:
        """
        # storages
        losses = tf.TensorArray(tf.float32, size=return_every_epochs)
        loss_cnt = 0

        # def get_batch():
        #     idxs = tf.range(dataset.shape[0])
        #     idxs = tf.random.shuffle(idxs)[:batch_size]
        #     return tf.gather(dataset, indices=idxs)

        for epoch_offset in tf.range(return_every_epochs):
            total_g_loss = total_d_loss = .0

            # training
            dsiter = iter(dataset)
            for i in tf.range(n_batch):
                for _ in tf.range(dg_train_ratio - 1):
                    self._train_step_discriminator(next(dsiter))
                    pass
                d_loss, g_loss = self._train_step_both(next(dsiter))
                total_d_loss += d_loss
                total_g_loss += g_loss

            # sample generated points
            total_d_loss /= n_batch
            total_g_loss /= n_batch
            losses = losses.write(loss_cnt, [total_d_loss, total_g_loss])
            loss_cnt += 1

        return losses.stack()
