import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from pathlib import Path
from data.datamaker import *
from utils.common import clean_images


class LSGAN:
    latent_factor = 5

    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.g_optimizer = tf.optimizers.Adam(0.001)
        self.d_optimizer = tf.optimizers.Adam(0.001)

        pass

    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(32, use_bias=False, input_shape=(self.latent_factor,)), layers.LeakyReLU(),
            layers.Dense(16), layers.LeakyReLU(),
            layers.Dense(2)
        ])
        return model

    def build_discriminator(self):
        model = keras.Sequential([
            layers.Dense(32, input_shape=(2,)), layers.LeakyReLU(),
            layers.Dense(16), layers.LeakyReLU(),
            layers.Dense(1)
        ])
        return model

    @tf.function
    def train_step(self, dataset, batch_size):
        def mse(y, pred):
            return tf.reduce_mean(tf.square(pred - y))  # over all axis

        true_labels, fake_labels = tf.ones([batch_size]), tf.zeros([batch_size])
        total_d_loss = total_g_loss = 0.0

        for x_real in dataset:
            noise = tf.random.normal([batch_size, self.latent_factor])
            with tf.GradientTape(persistent=True) as tape:
                x_fake = self.generator(noise, training=True)
                y_fake = self.discriminator(x_fake, training=True)
                y_real = self.discriminator(x_real, training=True)
                d_loss_fake = mse(y_fake, fake_labels)
                d_loss_real = mse(y_real, true_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2

                g_loss = mse(y_fake, true_labels)  # to generate real

            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)

            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

            total_d_loss += d_loss
            total_g_loss += g_loss

        return total_d_loss, total_g_loss

    def train(self, dataset, epochs, batch_size=64,
              sample_interval=10, image_saver=None):

        seed = tf.random.normal([300, self.latent_factor])
        dataset = dataset.batch(batch_size, drop_remainder=True)

        for epoch in range(epochs):
            start = time.time()

            total_d_loss, total_g_loss = self.train_step(dataset, batch_size)

            if epoch % sample_interval == 0 and image_saver is not None:
                image_saver(self.generator, epoch, seed)

            print(f'Epoch {epoch}/{epochs} cost {time.time() - start:.2f} s, '
                  f'G loss: {total_g_loss / len(dataset)}, D loss: {total_d_loss / len(dataset)}')


if __name__ == '__main__':
    save_path = 'pics/lsgan'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    clean_images(save_path)

    x, y, img_saver = make_cross_line_points(1024, 3, path=save_path)

    plt.scatter(x, y)
    plt.title(f'original data distribution')
    plt.savefig(f'{save_path}/dist.png')

    dataset = tf.data.Dataset.from_tensor_slices(np.array([x, y]).transpose().reshape((-1, 2)))

    lsgan = LSGAN()
    lsgan.train(dataset, 300, 32, image_saver=img_saver)
