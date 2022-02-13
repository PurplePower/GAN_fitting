import tensorflow as tf
import numpy as np
import sklearn
from sklearn.neighbors import KernelDensity

from models import BaseGAN
from metrics.BaseMetric import BaseMetric
from scipy.special import xlogy


class JensenShannonDivergence(BaseMetric):
    def __init__(self, bandwidth=0.05, generated_sample_size=500, mg_resolution=50, boundaries=None):
        super().__init__()
        self.bandwidth = bandwidth
        self.sample_size = generated_sample_size
        self.mg_resolution = mg_resolution
        self.boundaries = boundaries  # [(x_min, x_max), (y_min, y_max), ...]
        self.sample_points = None

        self.log_original_densities = None
        self.original_densities = None

    def estimate_densities(self, X, sample_points):
        kde = KernelDensity(bandwidth=self.bandwidth)
        kde.fit(X)
        log_probs = kde.score_samples(sample_points)
        probs = np.exp(log_probs)
        probs /= np.sum(probs)
        return probs

    def __call__(self, *args, **kwargs):
        model: BaseGAN = kwargs['model']
        dataset = kwargs['dataset']
        assert model and dataset is not None

        n_dim = dataset.shape[1]
        if self.boundaries is None:
            self.boundaries = [
                (np.min(dataset[:, i]) * 1.1, np.max(dataset[:, i]) * 1.1) for i in range(n_dim)]
            grid = np.meshgrid(*[
                np.linspace(b[0], b[1], self.mg_resolution)
                for b in self.boundaries
            ])
            self.sample_points = np.stack(list(map(lambda x: x.flatten(), grid)), axis=0).T

        # estimate original densities
        if self.log_original_densities is None:
            self.original_densities = self.estimate_densities(dataset, self.sample_points)
            # self.log_original_densities = np.log(self.original_densities)

        # estimate generated densities
        generated = model.generate(self.sample_size)
        gen_densities = self.estimate_densities(generated, self.sample_points)
        # log_gen_densities = np.log(gen_densities)

        # compute divergence
        m = (self.original_densities + gen_densities) / 2
        # log_m = np.log(m)
        # kl1 = np.sum(self.original_densities * (self.log_original_densities - log_m))
        # kl2 = np.sum(gen_densities * (log_gen_densities - log_m))
        kl1 = np.sum(xlogy(self.original_densities, self.original_densities / m))
        kl2 = np.sum(xlogy(gen_densities, gen_densities / m))

        jsd = (kl1 + kl2) / 2
        return jsd
