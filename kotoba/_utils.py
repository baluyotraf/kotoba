from itertools import filterfalse


def apply_to_nested(data, func=lambda x: x):
    try:
        iter(data)
        if isinstance(data, str):
            return func(data)
        else:
            return [apply_to_nested(i, func) for i in data]
    except TypeError:
        return func(data)


def uniquify(iterable):
    seen = set()
    seen_add = seen.add
    for element in filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element
