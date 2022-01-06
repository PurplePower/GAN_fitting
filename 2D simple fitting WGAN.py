import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
from data.datamaker import *
from utils.common import clean_images

# make D and G
latent_factor = 5


def get_generator():
    lr_g = 0.01
    return keras.Sequential([
        layers.Dense(32, use_bias=False, input_shape=(latent_factor,)), layers.LeakyReLU(),
        layers.Dense(16), layers.LeakyReLU(),
        layers.Dense(2)  # output 2D point
    ]), tf.keras.optimizers.SGD(lr_g)


def get_discriminator():
    lr_d = 0.01
    return keras.Sequential([
        layers.Dense(32, input_shape=(2,)), layers.LeakyReLU(),
        layers.Dense(16), layers.LeakyReLU(),
        layers.Dense(1)
    ]), tf.keras.optimizers.SGD(lr_d)


# loss functions

def gradient_penalty(D, x_real, x_fake):
    batchsz = x_real.shape[0]

    t = tf.random.uniform([batchsz, 1])
    interpolate = t * x_real
    interpolate += (1 - t) * x_fake

    with tf.GradientTape() as tape:
        tape.watch([interpolate])
        d_interpolate_logits = D(interpolate)

    grads = tape.gradient(d_interpolate_logits, interpolate)

    # get norm
    gp = tf.norm(tf.reshape(grads, [grads.shape[0], -1]), axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)
    return gp


# TODO: combine d_fake/real_logits in one
def discriminator_loss(D, G, z_noise, x_real, is_training):
    x_fake = G(z_noise, training=is_training)
    d_fake_logits = D(x_fake, training=is_training)
    d_real_logits = D(x_real, training=is_training)

    # gradient penalty
    gp = gradient_penalty(D, x_real, x_fake)

    loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 1.0 * gp
    return loss, gp


def generator_loss(D, G, z_noise, is_training):
    x_fake = G(z_noise)
    d_fake_logits = D(x_fake, training=is_training)
    loss = - tf.reduce_mean(d_fake_logits)
    return loss


# training step

@tf.function
def train_step(D, G, x_real, d_optimizer, g_optimizer):
    z_noise = tf.random.normal([len(x_real), latent_factor])

    # TODO: make each tape track only its Variables
    # TODO: may train D multiple times and G only once
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        disc_loss, gp = discriminator_loss(D, G, z_noise, x_real, True)  # disc_loss includes gp
        gen_loss = generator_loss(D, G, z_noise, True)

    disc_grads = d_tape.gradient(disc_loss, D.trainable_variables)
    gen_grads = g_tape.gradient(gen_loss, G.trainable_variables)

    d_optimizer.apply_gradients(zip(disc_grads, D.trainable_variables))
    g_optimizer.apply_gradients(zip(gen_grads, G.trainable_variables))

    return disc_loss, gen_loss


# run

def train(D, d_optimizer, G, g_optimizer, dataset: tf.data.Dataset, epochs, img_saver):
    seed = tf.random.normal([300, latent_factor])
    for epoch in range(epochs):
        start = time.time()

        if epoch % 20 == 0:
            img_saver(G, epoch, seed)

        total_gen_loss = total_disc_loss = 0.0

        for data_batch in dataset:
            disc_loss, gen_loss = train_step(D, G, data_batch, d_optimizer, g_optimizer)
            total_disc_loss += disc_loss
            total_gen_loss += gen_loss

        print(
            f'Epoch {epoch}/{epochs} cost {time.time() - start:.2f} s, '
            f'G loss: {total_gen_loss / len(dataset)}, D loss: {total_disc_loss / len(dataset)}'
        )

    img_saver(G, epochs - 1, seed)


if __name__ == '__main__':
    save_path = 'pics/wgan'
    x, y, img_saver = make_single_blob_points(1000,  path=save_path)
    # x, y, img_saver = make_cross_line_points(1000, 3, path=save_path)
    clean_images(save_path)

    plt.scatter(x, y)
    plt.title(f'original data distribution')
    plt.savefig(f'{save_path}/dist.png')
    # plt.show()

    G, g_optimizer = get_generator()
    D, d_optimizer = get_discriminator()

    dataset = tf.data.Dataset.from_tensor_slices(
        np.array([x, y], dtype=np.float32).transpose().reshape((-1, 2))).batch(16)

    train(D, d_optimizer, G, g_optimizer, dataset, 500, img_saver)

    pass
