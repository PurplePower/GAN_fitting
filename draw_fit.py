import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Adadelta
from tensorflow.keras.optimizers.schedules import ExponentialDecay, LearningRateSchedule

from metrics.JudgeSurface import JudgeSurface
from models import WGAN, KernelGAN, CustomStairBandwidth, SWG
from metrics import JensenShannonDivergence as JSD
from utils.structures import *
from utils.picdist import image2dots
from visualizers.ScatterSampler import ScatterSampler
from visualizers.plots import *
from utils.common import empty_directory

if __name__ == '__main__':

    save_path = './pics/draw'
    empty_directory(save_path)

    # read image and convert to dataset
    # canvas_length = 4.0
    # im = Image.open('pics/wgan_text.png')
    # im = im.resize((64, 64))

    canvas_length = 4.0
    im = Image.open('pics/南.png')
    im = im.resize((64, 64))
    x = image2dots(im, 1024, canvas_width=canvas_length, canvas_height=canvas_length)
    n_samples = x.shape[0]

    # plot data distribution
    sampler = ScatterSampler(save_path, 'draw fit')
    sampler.formats = ['png']
    sampler.xlim = sampler.ylim = (-(canvas_length / 2 * 1.1), canvas_length / 2 * 1.1)

    surface_sampler = JudgeSurface(path=save_path + '/surfaces')

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], 10)
    plt.title('Data Distribution')
    plt.savefig(f'{save_path}/dist.png')

    plt.clf()
    plt.hist2d(x[:, 0], x[:, 1], bins=100)
    plt.title('Data Density')
    plt.savefig(f'{save_path}/dist density.png')

    # build model and training
    latent_factor = 16
    D, G = level_5_structure(2, latent_factor)
    model_type = SWG

    if model_type is WGAN:
        epochs = 10000
        model = WGAN(
            2, latent_factor, D=D, G=G,
            d_optimizer=RMSprop(), g_optimizer=RMSprop()
            # d_optimizer=SGD(ExponentialDecay(1e-2, 4000, 0.1)),
            # g_optimizer=SGD(ExponentialDecay(1e-2, 4000, 0.1)),
            # d_optimizer=SGD(1e-2), g_optimizer=SGD(1e-2)
        )
        losses, metrics = model.train(
            x, epochs, batch_size=64,
            sample_number=1024, sampler=sampler, sample_interval=20,
            dg_train_ratio=5, metrics=[JSD(), surface_sampler]
        )
    elif model_type is KernelGAN:
        exp = ExponentialDecay(0.10, 1500, 0.2)


        class MyBandwidth(LearningRateSchedule):
            def __init__(self, exp, decrease_start=100, init_bw=0.3):
                self.exp = exp
                self.decrease_start = decrease_start
                self.init_bw = init_bw

            def get_config(self):
                return {
                    'name': self.__class__.__name__,
                    'exp': self.exp.get_config(),
                    'decrease_start': self.decrease_start,
                    'init_bw': self.init_bw
                }

            @classmethod
            def from_config(cls, config):
                exp = ExponentialDecay.from_config(config['exp'])
                return MyBandwidth(exp, config['decrease_start'], config['init_bw'])

            def __call__(self, step, **kwargs):
                if step < self.decrease_start:
                    return self.init_bw
                else:
                    return self.exp(step - self.decrease_start)


        # bw_updater = MyBandwidth(exp)
        bw_updater = CustomStairBandwidth([(500, 0.5), (10000, 0.25), (20000, 0.125), (30000, 0.05)], 0.05)

        model = KernelGAN(
            2, latent_factor, D=None, G=G,
            g_optimizer=Adadelta(1),
            # g_optimizer=SGD(1e-2),
            bandwidth=0.05, bandwidth_updater=bw_updater
        )
        losses, metrics = model.train(
            x, 40000, batch_size=1024,
            sample_number=512, sampler=sampler, sample_interval=1000,
            metrics=[JSD(), surface_sampler]
        )
    elif model_type is SWG:
        model = SWG(
            2, latent_factor, D=D, G=G,
            d_optimizer=Adam(1e-4), g_optimizer=Adam(1e-4),
            n_directions=256,
            use_discriminator=True, lambda1=1.0
        )

        losses, metrics = model.train(
            x, 10000, batch_size=128, sample_interval=50,
            sampler=sampler, sample_number=1024,
            metrics=[JSD(), surface_sampler]
        )
    else:
        raise Exception()

    model.save(save_path + '/model')

    # plots
    plt.figure()
    plt.plot(losses)
    plt.title('Losses over training epochs')
    plt.legend(['D losses', 'G losses'])
    plt.savefig(f'{save_path}/losses.png')
    plt.savefig(f'{save_path}/losses.svg')

    plt.figure()
    plt.plot(metrics[0])
    plt.title('JSD')
    plt.savefig(f'{save_path}/jsd.png')
    plt.savefig(f'{save_path}/jsd.svg')

    plt.figure('2D Density')
    plot_2d_density(model, 256 * 1024)
    plt.savefig(f'{save_path}/density.png')
    plt.savefig(f'{save_path}/density.svg')

    plt.figure()
    plot_2d_discriminator_judge_area(model, x, projection='3d')
    plt.savefig(f'{save_path}/judge area 3D.png')
    plt.savefig(f'{save_path}/judge area 3D.svg')

    plt.clf()
    plot_2d_discriminator_judge_area(model, x, projection='2d')
    plt.savefig(f'{save_path}/judge area.png')
    plt.savefig(f'{save_path}/judge area.svg')

    plt.show()
    pass
