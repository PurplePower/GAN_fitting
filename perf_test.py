import tensorflow as tf
import numpy as np
import time


# ds = tf.data.Dataset.from_tensor_slices(np.ones(1024).astype(np.float32)).batch(16)
# n_iter = 10
#
# @tf.function
# def small_step(x_batch):
#     return tf.reduce_sum(x_batch)
#
#
# def small_loop(ds):
#     for i in range(n_iter):
#         s = 0.0
#         for x in ds:
#             s += small_step(x)
#
#     pass
#
#
# #############
#
# @tf.function
# def large_loop(ds):
#     for i in range(n_iter):
#         s = 0.0
#         for x in ds:
#             s += tf.reduce_sum(x)
#
#
# # small loop
# for i in range(10):
#     start = time.time()
#     small_loop(ds)
#     print(f'small loop cost {time.time() - start:.2f} s')
#
# for i in range(10):
#     start = time.time()
#     large_loop(ds)
#     print(f'large loop cost {time.time() - start:.2f} s')


class TF_Numpy_tester():
    def __init__(self):
        self.model = None
        self.input_size = 8
        self.opt = tf.keras.optimizers.SGD()
        self.reset_model()

    def reset_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.input_size,)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(1)
        ])

    def test(self):
        n_sample = 1024
        x_np = np.random.normal(0, 1, size=(n_sample, self.input_size))
        x_tf = tf.random.normal([n_sample, self.input_size])

        for i in range(20):
            print(f'Test round {i + 1}')

            start = time.time()
            self.test_on_array(x_np)
            print(f'Model with numpy array costs {time.time() - start:.4f} s')
            self.reset_model()

            start = time.time()
            self.test_on_array(x_tf)
            print(f'Model with tf.Tensor costs {time.time() - start:.4f} s')
            self.reset_model()

        pass

    @tf.function
    def test_on_array(self, x):
        for epoch in range(10):
            with tf.GradientTape() as tape:
                outs = self.model(x, training=True)
                loss = tf.reduce_mean(outs - 1.0, axis=1)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        pass


if __name__ == '__main__':
    tester = TF_Numpy_tester()
    tester.test()

    pass
