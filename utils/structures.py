import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LeakyReLU, GaussianNoise, BatchNormalization


def _get_dense(units, n_layers):
    layers = []
    for i in range(n_layers):
        layers.append(Dense(units))
        layers.append(LeakyReLU())
    return layers


def level_1_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(32, input_shape=(input_dim,)), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(32, input_shape=(latent_factor,)), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_1_bn_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(32, input_shape=(input_dim,)), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(32, input_shape=(latent_factor,)), BatchNormalization(), LeakyReLU(),
        Dense(16), BatchNormalization(), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_2_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(32, input_shape=(input_dim,)), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(32, input_shape=(latent_factor,)), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_2_bn_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(32, input_shape=(input_dim,)), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(32, input_shape=(latent_factor,)), BatchNormalization(), LeakyReLU(),
        Dense(32), BatchNormalization(), LeakyReLU(),
        Dense(16), BatchNormalization(), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_2_noise_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(32, input_shape=(input_dim,)), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(16), LeakyReLU(),
        GaussianNoise(1 / 16),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(32, input_shape=(latent_factor,)), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(16), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_3_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(128, input_shape=(input_dim,)), LeakyReLU(),
        Dense(64), LeakyReLU(),
        Dense(64), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(128, input_shape=(latent_factor,)), LeakyReLU(),
        Dense(64), LeakyReLU(),
        Dense(64), LeakyReLU(),
        Dense(32), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_3a_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(128, input_shape=(input_dim,)), LeakyReLU(),
        Dense(128), LeakyReLU(),
        Dense(128), LeakyReLU(),
        Dense(64), LeakyReLU(),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(128, input_shape=(latent_factor,)), LeakyReLU(),
        Dense(128), LeakyReLU(),
        Dense(128), LeakyReLU(),
        Dense(64), LeakyReLU(),
        Dense(input_dim)
    ])
    return D, G


def level_4_structure(input_dim, latent_factor):
    D = keras.Sequential([
        Dense(64, input_shape=(input_dim,)), LeakyReLU(),
        *_get_dense(128, 6),
        Dense(1)
    ])
    G = keras.Sequential([
        Dense(128, input_shape=(latent_factor,)), LeakyReLU(),
        *_get_dense(128, 6),
        Dense(input_dim)
    ])
    return D, G


STRUCTURE_NAMES = {
    level_1_structure: '1',
    level_1_bn_structure: '1_bn',
    level_2_structure: '2',
    level_2_bn_structure: '2_bn',
    level_2_noise_structure: '2_noise',
    level_3_structure: '3',
    level_3a_structure: '3a',
    level_4_structure: '4'
}
