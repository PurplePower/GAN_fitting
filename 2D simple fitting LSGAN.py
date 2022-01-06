import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from pathlib import Path
from data.datamaker import *
from utils.common import clean_images

####################
#  set a=0, b=c=1


latent_factor = 5


def get_generator():
    model = keras.Sequential([
        layers.Dense(32, use_bias=False, input_shape=(latent_factor,)), layers.LeakyReLU(),
        layers.Dense(16), layers.LeakyReLU(),
        layers.Dense(2)
    ])
    # optimizer = keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(optimizer, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


def get_discriminator():
    model = keras.Sequential([
        layers.Dense(32, input_shape=(2,)), layers.LeakyReLU(),
        layers.Dense(16), layers.LeakyReLU(),
        layers.Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss='mse')
    return model


def train(D, G, dataset, epochs, batch_size=64, sample_interval=10, img_saver=None):
    seed = tf.random.normal([300, latent_factor])
    # get combined model
    input_layer = layers.Input((latent_factor,))
    faked = G(input_layer)
    D.trainable = False
    label = D(faked)
    combined = keras.Model(input_layer, label)
    combined.compile(loss='mse', optimizer=tf.optimizers.Adam(0.001))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    true_labels, fake_labels = tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])
    labels = tf.concat([true_labels, fake_labels], 0)

    for epoch in range(epochs):
        start = time.time()

        total_d_loss = total_g_loss = 0.0
        for x_real in dataset:
            noise = tf.random.normal([batch_size, latent_factor])
            # coding for true, a=1; coding for fake, b=c=0

            ###############################
            #  train discriminator
            ###############################
            # x_fake = G(noise, training=False)  # can't use predict
            x_fake = G.predict(noise)

            # TODO: combine as one batch for a single train
            # d_loss_real = D.train_on_batch(x_real, true_labels)
            # d_loss_fake = D.train_on_batch(x_fake, fake_labels)
            # d_loss = (d_loss_real + d_loss_fake) / 2

            d_loss = 0.5 * D.train_on_batch(tf.concat([x_real, x_fake], 0), labels)

            ###############################
            #  train generator
            ###############################
            g_loss = combined.train_on_batch(noise, true_labels)

            total_d_loss += d_loss
            total_g_loss += g_loss

        if epoch % sample_interval == 0:
            if img_saver is not None:
                img_saver(G, epoch, seed)

        print(f'Epoch {epoch}/{epochs} cost {time.time() - start:.2f} s, '
              f'G loss: {total_g_loss / len(dataset)}, D loss: {total_d_loss / len(dataset)}')

    pass


if __name__ == '__main__':
    save_path = 'pics/lsgan'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    clean_images(save_path)

    x, y, img_saver = make_cross_line_points(1000, 3, path=save_path)

    plt.scatter(x, y)
    plt.title(f'original data distribution')
    plt.savefig(f'{save_path}/dist.png')

    dataset = tf.data.Dataset.from_tensor_slices(np.array([x, y]).transpose().reshape((-1, 2)))

    D, G = get_discriminator(), get_generator()

    train(D, G, dataset, 300, batch_size=32, img_saver=img_saver)
