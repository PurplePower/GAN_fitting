import numpy as np
import tensorflow as tf
from models import GAN, LSGAN, WGAN
from data.datamaker import make_ring_dots

if __name__ == '__main__':
    model_types = [GAN, LSGAN, WGAN]
    x, sampler = make_ring_dots(128)  # for simple test
    dataset = tf.data.Dataset.from_tensor_slices(x)


    for cls in model_types:
        path = f'./tmp/{cls.__name__}'
        print(f'Testing {cls.__name__}...')
        model = cls(input_dim=2)
        x_test = np.random.normal(0, 1, size=(4, model.latent_factor))

        model.train(dataset, 5, sampler=None)

        # keep current weights
        cur_weights = [
            model.discriminator.weights, model.generator.weights,
            model.d_optimizer.weights, model.g_optimizer.weights
        ]
        cur_pred = model.generate(seed=x_test)

        model.save(path)
        del model

        # load new model
        new_model = cls.load(path)
        new_weights = [
            new_model.discriminator.weights, new_model.generator.weights,
            new_model.d_optimizer.weights, new_model.g_optimizer.weights
        ]

        # compare weights
        for cur, new in zip(cur_weights, new_weights):
            if isinstance(cur, list):
                if len(cur) != len(new):
                    raise Exception('Weights length not the same')
                for w1, w2 in zip(cur, new):
                    assert np.alltrue(w1 == w2)
            else:
                assert np.alltrue(cur == new)

        # compare predict results
        assert np.alltrue(cur_pred == new_model.generate(seed=x_test))

        print(f'Class {cls.__name__} test passed!')

    # clean the tmp
    pass
