import matplotlib.pyplot as plt
from pathlib import Path

from metrics import BaseMetric
from visualizers.plots import plot_2d_discriminator_judge_area, determine_boundaries


class JudgeSurface(BaseMetric):
    def __init__(self, path, resolution=100):
        super(JudgeSurface, self).__init__()
        plt.ioff()
        self.path = path
        self.figure = plt.figure()
        self.resolution = resolution
        self.boundaries = None

        Path(path).mkdir(parents=True, exist_ok=True)

    def __call__(self, model, dataset, epoch, *args, **kwargs):
        plt.clf()
        self.boundaries = self.boundaries or determine_boundaries(dataset, .5, .5)
        plot_2d_discriminator_judge_area(
            model, dataset, self.boundaries, self.resolution)
        plt.title(f'D surface at epoch {epoch}')
        plt.savefig(f'{self.path}/surface_{epoch}.png')
        pass
