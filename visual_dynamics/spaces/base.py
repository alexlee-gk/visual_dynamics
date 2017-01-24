from visual_dynamics.utils.config import ConfigObject


class Space(ConfigObject):
    """
    Provides a classification state spaces and action spaces,
    so you can write generic code that applies to any Environment.
    E.g. to choose a random action.
    """

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def clip(self, x, out=None):
        raise NotImplementedError

    @staticmethod
    def create(other):
        """
        Creates a space from another space-like object
        compatible with this class
        """
        from visual_dynamics import spaces
        return getattr(spaces, other.__class__.__name__).create(other)
