from spaces import Space


class TupleSpace(Space):
    """
    A tuple (i.e., product) of simpler spaces
    Example usage:
    self.observation_space = spaces.TupleSpace((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space, part) in zip(self.spaces, x))

    def clip(self, x, out=None):
        if out is None:
            out = [None] * len(self.spaces)
        else:
            assert isinstance(out, (tuple, list))
            assert len(out) == len(self.spaces)
        return tuple([space.clip(x, out=out) for (space, x, out) in zip(self.spaces, x, out)])

    @property
    def shape(self):
        return tuple([space.shape() for space in self.spaces])

    def __eq__(self, other):
        return isinstance(other, (tuple, list)) and \
               len(other) == len(self.spaces) and \
               all([(space == other_space) for (space, other_space) in zip(self.spaces, other.spaces)])

    def _get_config(self):
        config = super(TupleSpace, self)._get_config()
        config.update({'spaces': self.spaces})
        return config

