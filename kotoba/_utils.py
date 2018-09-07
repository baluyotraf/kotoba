from itertools import filterfalse, chain


def _is_iterable(iterable):
    try:
        iter(iterable)
        return not isinstance(iterable, str)
    except TypeError:
        return False


def _as_iterable(element):
    yield element


def map_elements(data, func=lambda x: x):
    if _is_iterable(data):
        return [map_elements(d, func) for d in data]
    else:
        return func(data)


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
