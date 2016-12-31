from spaces import Space


class DictSpace(Space):
    """
    Like TupleSpace except that each of the spaces have a name associated with it
    """
    def __init__(self, spaces):
        if not isinstance(spaces, dict):
            raise ValueError('spaces should be a dict but it is a %r' % type(spaces))
        self.spaces = spaces

    def sample(self):
        return dict([(key, space.sample()) for (key, space) in self.spaces.items()])

    def contains(self, x):
        return isinstance(x, dict) and set(x.keys()) == set(self.spaces.keys()) and all(
            space.contains(x[key]) for (key, space) in self.spaces.items())

    def clip(self, x, out=None):
        if out is None:
            out = dict([(key, None) for key in self.spaces.keys()])
        else:
            assert isinstance(out, dict)
            assert set(out.keys()) == set(self.spaces.keys())
        return dict([(key, space.clip(x[key], out=out[key])) for (key, space) in self.spaces.items()])

    @property
    def shape(self):
        return dict([(key, space.shape()) for (key, space) in self.spaces.items()])

    def __eq__(self, other):
        return isinstance(other, DictSpace) and \
               set(other.spaces.keys()) == set(self.spaces.keys()) and \
               all([(space == other.spaces[key]) for (key, space) in self.spaces.items()])

    def _get_config(self):
        config = super(DictSpace, self)._get_config()
        config.update({'spaces': self.spaces})
        return config
