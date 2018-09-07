from abc import ABCMeta, abstractmethod
from ._utils import map_elements


class Preprocessor(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, x):
        pass


class Pipeline(Preprocessor):

    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def transform(self, x):
        for p in self._preprocessors:
            x = p.transform(x)
        return x


class FunctionPreprocessor(Preprocessor):

    def __init__(self, func, as_iterable=False):
        self._func = func
        self._as_iterable = as_iterable

    def transform(self, x):
        return map_elements(x, self._func, self._as_iterable)


class LowerCase(FunctionPreprocessor):

    def __init__(self, as_iterable=False):
        super().__init__(lambda x: x.lower(), as_iterable)
