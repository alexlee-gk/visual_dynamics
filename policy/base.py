import utils


class Policy(utils.config.ConfigObject):
    def act(self, obs):
        """
        Returns the action of this policy given the observations, and this
        action may be non-deterministic.

        Args:
            obs: tuple or list of observations

        Returns:
            the action of this policy
        """
        raise NotImplementedError

    def reset(self):
        """
        Returns:
            the reset state of this policy
        """
        raise NotImplementedError
