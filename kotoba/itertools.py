from itertools import filterfalse, chain, islice
from collections.abc import Mapping


_is_iterable_exceptions = [str, bytes, Mapping]


def is_iterable(iterable):
    """
    Checks if an object is an iterable
    :param iterable: object to check
    :return: True if iterable else False
    """
    try:
        iter(iterable)
        for t in _is_iterable_exceptions:
            if isinstance(iterable, t):
                return False
        return True
    except TypeError:
        return False


def as_iterable(element):
    """
    Converts a single element to a generator
    :param element: element to convert
    :return: generator with the given element
    """
    yield element


def map_iterable(data, func):
    """
    Applies a function to any iterable and to its iterable elements
    :param data: iterable or object to apply the function
    :param func: function used for mapping
    :return:
    """
    if is_iterable(data):
        return func((map_iterable(d, func) for d in data))
    else:
        return data


def _map_elements(data, func):
    if is_iterable(data):
        return (_map_elements(d, func) for d in data)
    else:
        return func(data)


def map_elements(data, func, as_iterable=False):
    """
    Applies a function to the non-iterable items on all iterable in data
    :param data: iterable or object to apply the function
    :param func: function used for mapping
    :param as_iterable: defines if the result must be an iterable or a list
    :return:
    """
    mapped = _map_elements(data, func)
    if not as_iterable:
        mapped = map_iterable(mapped, list)
    return mapped


def flatten(data):
    """
    Collapses all iterable items into a single iterable
    :param data: iterable or object to collapse
    :return: flattened iterable
    """
    if is_iterable(data):
        return chain.from_iterable((flatten(d) for d in data))
    else:
        return as_iterable(data)


def uniquify(iterable):
    """
    Removes duplicate elements in an iterable while preserving order
    :param iterable: iterable find the unique elements
    :return: iterable of the unique elements
    """
    seen = set()
    seen_add = seen.add
    for element in filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element


def _batch(iterable, n):
    iterator = iter(iterable)
    while True:
        current_batch = list(islice(iterator, n))
        if current_batch:
            yield current_batch
        else:
            break


def batch(iterable, n, as_iterable=False):
    """
    Groups an iterable into an iterable of n elements
    :param iterable: iterable to batch
    :param n: number of grouped elements
    :param as_iterable: defines if the result must be an iterable or a list
    :return: batched data iterable
    """
    batched = _batch(iterable, n)
    if not as_iterable:
        batched = map_iterable(batched, list)
    return batched


def map_last_dim(iterable, func, as_iterable=False):
    """
    Applies the function only to the last dimensions of the iterable
    :param iterable: iterable to map the last dimension
    :param func: function for mapping
    :param as_iterable: defines if the result must be an iterable or a list
    :return: iterable of the mapped data
    """
    if is_iterable(iterable):
        iter_list = list(iterable)
        if any((is_iterable(element) for element in iter_list)):
            mapped = (map_last_dim(element, func, as_iterable)
                      for element in iter_list)
            if not as_iterable:
                mapped = list(mapped)
            return mapped
        else:
            return func(iter_list)
    else:
        return iterable
