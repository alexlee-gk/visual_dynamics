from __future__ import division, print_function
import numpy as np
from spaces import Space


class ConcatenationSpace(Space):
    def __init__(self, spaces):
        self.spaces = spaces
        sizes = []
        for space in self.spaces:
            size, = space.shape
            sizes.append(size)
        cumsum = np.cumsum(sizes)
        self.slices = [slice(start, stop) for start, stop in zip((0,) + tuple(cumsum[:-1]), cumsum)]

    def sample(self):
        return np.concatenate([space.sample() for space in self.spaces])

    def contains(self, x):
        assert x.shape == self.shape
        return all(space.contains(x[s]) for space, s in zip(self.spaces, self.slices))

    def clip(self, x, out=None):
        assert x.shape == self.shape
        if out is not None:
            assert out.shape == self.shape
        return np.concatenate([space.clip(x[s], out=(out[s] if out is not None else None))
                               for space, s in zip(self.spaces, self.slices)])

    @property
    def shape(self):
        return (self.slices[-1].stop,)

    def _get_config(self):
        config = super(ConcatenationSpace, self)._get_config()
        config.update({'spaces': self.spaces})
        return config

    @staticmethod
    def create(other):
        return ConcatenationSpace(other.spaces)
