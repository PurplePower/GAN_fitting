import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob

# make 1D dataset
DATASET_SIZE = 1200
# data = tf.data.Dataset.from_tensor_slices(np.random.normal(4, 1, (DATASET_SIZE, 1)))
data = tf.data.Dataset.from_tensor_slices(np.concatenate([
    np.random.normal(-4, 0.81, DATASET_SIZE // 2),
    np.random.normal(4, 0.81, DATASET_SIZE // 2)
]).reshape((-1, 1)))


files = glob.glob('pics/*.png')
for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))


# plot real data distribution
plt.figure()
plt.hist(np.reshape(list(data.as_numpy_iterator()), (-1,)))
plt.title(f'Data to imitate')
plt.savefig('pics/dist.png')
# plt.show()

# normalization...

BATCH_SIZE = 256
NOISE_DIM = 10
lr_g = 1e-4
lr_d = 1e-4

data = data.shuffle(DATASET_SIZE).batch(BATCH_SIZE)
seed = tf.random.normal([100, NOISE_DIM])

###########################
# models
###########################


def make_generator():
    """
    Make a simple generator
    :return:
    """
    model = keras.Sequential()
    # L1
    model.add(layers.Dense(128, use_bias=False, input_shape=(NOISE_DIM,)))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # L2
    model.add(layers.Dense(64))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # L3
    # model.add(layers.Dense(64))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # L4, output 1D point
    model.add(layers.Dense(1))

    return model


def make_discriminator():
    """
    Make a simple discriminator
    :return:
    """

    return keras.Sequential([
        layers.Dense(128, input_shape=(1,)),
        layers.LeakyReLU(),
        # layers.Dense(128),
        # layers.LeakyReLU(),
        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dense(1)     # log(probability)
    ])

#######################
#  loss
#######################


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)    # Discriminator outputs logits


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


######################
# training
######################

generator = make_generator()
discriminator = make_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(lr_g)
discriminator_optimizer = tf.keras.optimizers.Adam(lr_d)


@tf.function
def train_step(real_x):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generate fake data
        fake_x = generator(noise, training=True)

        real_output = discriminator(real_x, training=True)
        fake_output = discriminator(fake_x, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # update gradient
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, input_noise):
    generated_data = model(input_noise, training=False)
    plt.figure()
    plt.hist(np.array(generated_data).reshape((-1,)))
    plt.title(f'Generated data at epoch {epoch}')
    plt.savefig(f'pics/data_epoch_{epoch}.png')
    # plt.show(block=False)


def train(dataset: tf.data.Dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = total_disc_loss = 0.0

        for data_batch in dataset:
            gen_loss, disc_loss = train_step(data_batch)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss

        print(f'Epoch {epoch}/{epochs} cost {time.time()-start:.2f} s, G loss: {total_gen_loss}, D loss: {total_disc_loss}')

        if (epoch + 1) % 20 == 0:
            generate_and_save_images(generator, epoch, seed)


if __name__ == '__main__':
    train(data, 500)
    pass



