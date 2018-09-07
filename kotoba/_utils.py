from itertools import filterfalse, chain


def _is_iterable(iterable):
    try:
        iter(iterable)
        return not isinstance(iterable, str)
    except TypeError:
        return False


def _as_iterable(element):
    yield element


def map_iterable(data, func=lambda x: x):
    if _is_iterable(data):
        return func((map_iterable(d, func) for d in data))
    else:
        return data


def _map_elements(data, func=lambda x: x):
    if _is_iterable(data):
        return (_map_elements(d, func) for d in data)
    else:
        return func(data)


def map_elements(data, func=lambda x: x, as_list=True):
    mapped = _map_elements(data, func)
    if as_list:
        mapped = map_iterable(mapped, list)
    return mapped


def flatten(data):
    if _is_iterable(data):
        return chain.from_iterable((flatten(d) for d in data))
    else:
        return _as_iterable(data)


def uniquify(iterable):
    seen = set()
    seen_add = seen.add
    for element in filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element
