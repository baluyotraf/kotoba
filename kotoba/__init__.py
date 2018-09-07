from abc import ABCMeta, abstractmethod
from ._utils import map_elements
from . import embedding


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

    def __init__(self, func):
        self._func = func

    def transform(self, x):
        return map_elements(x, self._func)


class LowerCase(FunctionPreprocessor):

    def __init__(self):
        super().__init__(lambda x: x.lower())
