import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from data.datamaker import *
from utils.common import clean_images

# params
latent_factor = 4
lr_g, lr_d = 1e-3, 1e-3

G = keras.Sequential([
    layers.Dense(32, use_bias=False, input_shape=(latent_factor,)), layers.LeakyReLU(),
    layers.Dense(16), layers.LeakyReLU(),
    layers.Dense(2)  # output 2D point
])

D = keras.Sequential([
    layers.Dense(32, input_shape=(2,)), layers.LeakyReLU(),
    layers.Dense(16), layers.LeakyReLU(),
    layers.Dense(1)
])

g_optimizer = tf.keras.optimizers.Adam(lr_g)
d_optimizer = tf.keras.optimizers.Adam(lr_d)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


@tf.function
def train_step(generator: keras.Sequential, discriminator: keras.Sequential, real_x):
    print('Tracing...')  # tf.function trace for only a few times
    noise = tf.random.normal([len(real_x), latent_factor])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_x = generator(noise, training=True)

        real_output = discriminator(real_x, training=True)
        fake_output = discriminator(fake_x, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # update gradient
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(generator, discriminator, dataset: tf.data.Dataset, epochs, img_saver):
    seed = tf.random.normal([300, latent_factor])
    for epoch in range(epochs):
        start = time.time()
        if epoch % 20 == 0:
            img_saver(generator, epoch + 1, seed)

        total_gen_loss = total_disc_loss = 0.0

        for data_batch in dataset:
            gen_loss, disc_loss = train_step(generator, discriminator, data_batch)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print(
            f'Epoch {epoch}/{epochs} cost {time.time() - start:.2f} s, '
            f'G loss: {total_gen_loss / len(dataset)}, D loss: {total_disc_loss / len(dataset)}')

    img_saver(generator, epochs, seed)


if __name__ == '__main__':
    save_path = 'pics/gan'
    # x, y, img_saver = make_line_points(1000, -2, 0, path=save_path)
    x, y, img_saver = make_cross_line_points(1000, 3, path=save_path)

    # x, y, img_saver = make_single_blob_points(1000)
    clean_images(save_path)

    plt.scatter(x, y)
    plt.title(f'original data distribution')
    plt.savefig(f'{save_path}/dist.png')
    # plt.show()

    # better with 16 batch size
    dataset = tf.data.Dataset.from_tensor_slices(np.array([x, y]).transpose().reshape((-1, 2))).batch(16)

    train(G, D, dataset, 300, img_saver)

    pass
