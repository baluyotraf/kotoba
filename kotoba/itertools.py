from itertools import filterfalse, chain, islice


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


def map_elements(data, func=lambda x: x, as_iterable=False):
    mapped = _map_elements(data, func)
    if not as_iterable:
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


def _batch(iterable, n):
    iterator = iter(iterable)
    while True:
        current_batch = list(islice(iterator, n))
        if current_batch:
            yield current_batch
        else:
            break


def batch(iterable, n, as_iterable=False):
    batched = _batch(iterable, n)
    if not as_iterable:
        batched = map_iterable(batched, list)
    return batched


def map_last_dim(iterable, func, as_iterable=False):
    if _is_iterable(iterable):
        iter_list = list(iterable)
        if any((_is_iterable(element) for element in iter_list)):
            mapped = (map_last_dim(element, func, as_iterable)
                      for element in iter_list)
            if not as_iterable:
                mapped = list(mapped)
            return mapped
        else:
            return func(iter_list)
    else:
        return iterable