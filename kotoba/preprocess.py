from abc import ABCMeta, abstractmethod
from .itertools import map_elements


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


class FunctionPreprocessor(Preprocessor):

    def __init__(self, func):
        self._func = func

    def transform(self, x, as_iterable=False):
        return map_elements(x, self._func, as_iterable)


class LowerCase(FunctionPreprocessor):

    def __init__(self):
        super().__init__(lambda x: x.lower())
