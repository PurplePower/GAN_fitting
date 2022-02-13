import time

import tensorflow as tf
import numpy as np
from typing import Union
from models import GAN
from visualizers.BaseSampler import BaseSampler


class GAN_Tensor(GAN):

    def train(self, dataset: Union[tf.Tensor, np.ndarray], epochs, batch_size=32, sample_interval=20,
              sampler: BaseSampler = None, sample_number=300, metrics=[],
              dg_train_ratio=1):
        if isinstance(dataset, np.ndarray):
            dataset = tf.constant(dataset)
        elif not isinstance(dataset, tf.Tensor):
            raise Exception(f'Currently not supported dataset as {type(dataset)}')
        seed = tf.random.normal([sample_number, self.latent_factor])
        n_samples = dataset.shape[0]
        n_batch = int(np.ceil(n_samples / batch_size))
        losses, metric_values = [], [[] for m in metrics]

        def get_batch():  # help get infinite batches by looping the dataset
            ds = tf.random.shuffle(dataset)
            while True:
                for batch_no in range(n_batch):
                    yield ds[(t := batch_no * batch_size):t + batch_size]
                ds = tf.random.shuffle(ds)

        batch_getter = get_batch()

        for epoch in range(epochs):
            start = time.time()
            kwargs = {
                'generator': self.generator, 'discriminator': self.discriminator,
                'model': self, 'dataset': dataset
            }
            total_d_loss = total_g_loss = .0

            # in each batch, train D for dg_train_ratio times and G once
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
                pass

            total_g_loss /= n_batch
            total_d_loss /= n_batch
            losses.append((total_d_loss, total_g_loss))
            self.print_epoch(epoch, epochs, time.time() - start, total_d_loss, total_g_loss)

        self.trained_epoch += epochs

        return np.array(losses), np.array(metric_values)
