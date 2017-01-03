import numpy as np
from spaces import Space


class BoxSpace(Space):
    """
    A box in R^n or Z^n.
    I.e., each coordinate is bounded.

    Example usage:
    self.action_space = spaces.BoxSpace(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low, high, shape=None, dtype=None):
        """
        Two kinds of valid input:
            BoxSpace(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            BoxSpace(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        self._shape = shape
        dtype = np.dtype(dtype)
        if shape is None:
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            assert self.low.shape == self.high.shape
        else:
            assert (np.isscalar(low) or low.ndim == 0) and \
                   (np.isscalar(high) or high.ndim == 0)
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)

    def sample(self):
        if np.issubdtype(self.dtype, int):
            return np.random.random_integers(low=self.low, high=self.high, size=self.shape)
        else:
            return np.random.uniform(low=self.low, high=self.high, size=self.shape)

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    def clip(self, x, out=None):
        return np.clip(x, self.low, self.high, out=out)

    @property
    def shape(self):
        if self._shape is None:
            return self.low.shape
        else:
            return self._shape

    @property
    def dtype(self):
        return self.low.dtype

    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

    def _get_config(self):
        config = super(BoxSpace, self)._get_config()
        config.update({'low': self.low.tolist(),
                       'high': self.high.tolist(),
                       'shape': self._shape,
                       'dtype': self.dtype.name})
        return config

    @staticmethod
    def create(other):
        return BoxSpace(other.low, other.high, shape=other._shape, dtype=other.dtype)
