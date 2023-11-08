from functools import reduce


def flat_map(f, xs):
    return reduce(lambda a, b: a + list(b), map(f, xs), [])


def flatten(xs):
    return flat_map(lambda x: x, xs)

def compose(*fs):
    def composed(x):
        for f in fs:
            x = f(x)
        return x

    return composed
