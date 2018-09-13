from abc import ABCMeta, abstractmethod
from .itertools import map_elements, map_last_dim, batch, map_iterable


class Preprocessor(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, x, as_iterable=False):
        pass


class Pipeline(Preprocessor):

    def __init__(self, preprocessors):
        self._preprocessors = preprocessors[:-1]
        self._final_preprocessor = preprocessors[-1]

    def transform(self, x, as_iterable=False):
        for p in self._preprocessors:
            x = p.transform(x, as_iterable=True)
        x = self._final_preprocessor.transform(x, as_iterable)
        return x


class HorizontalPipeline(Preprocessor):

    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def _transform(self, x, as_iterable):
        for (d, p) in zip(x, self._preprocessors):
            yield p.transform(d, as_iterable)

    def transform(self, x, as_iterable=False):
        piped = self._transform(x, as_iterable)
        if not as_iterable:
            piped = list(piped)
        return piped


class Raw(Preprocessor):

    def __init__(self, func):
        self._func = func

    def transform(self, x, as_iterable=False):
        try:
            return self._func(x, as_iterable)
        except TypeError:
            return self._func(x)


class MapItems(Preprocessor):

    def __init__(self, func):
        self._func = func

    def transform(self, x, as_iterable=False):
        return map_elements(x, self._func, as_iterable)


class LowerCase(MapItems):

    def __init__(self):
        super().__init__(lambda x: x.lower())


class Transpose2D(Preprocessor):

    def transform(self, x, as_iterable=False):
        transposed = zip(*x)
        if not as_iterable:
            transposed = map_iterable(transposed, list)
        return transposed


class Batch(Preprocessor):

    def __init__(self, n):
        self._n = n

    def transform(self, x, as_iterable=False):
        return batch(x, self._n, as_iterable)
