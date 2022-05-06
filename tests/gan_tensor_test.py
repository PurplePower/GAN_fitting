import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

from models import GAN_Tensor
from data.datamaker import make_ring_dots
from utils.structures import *

if __name__ == '__main__':
    save_path = './pics/gan_tensor'
    x, sampler = make_ring_dots(1024, path=save_path)

    D, G = level_1_structure(2, 5)

    model = GAN_Tensor(
        2, D=D, G=G,
        d_optimizer=SGD(1e-3), g_optimizer=SGD(1e-3)
    )

    # training
    losses, metrics = model.train(
        x, 400, 64,
        sample_interval=50, sampler=sampler, sample_number=512,
        dg_train_ratio=1
    )

    plt.figure()
    plt.plot(losses)
    plt.legend(['D loss', 'G loss'])
    plt.show()

    pass
