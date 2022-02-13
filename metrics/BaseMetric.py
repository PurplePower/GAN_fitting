import abc


class BaseMetric(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
    