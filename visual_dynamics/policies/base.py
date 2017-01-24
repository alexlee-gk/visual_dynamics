from visual_dynamics.utils.config import ConfigObject


class Policy(ConfigObject):
    def act(self, obs):
        """
        Returns the action of this policies given the observations, and this
        action may be non-deterministic.

        Args:
            obs: tuple or list of observations

        Returns:
            the action of this policies
        """
        raise NotImplementedError

    def reset(self):
        """
        Returns:
            the reset state of this policies
        """
        raise NotImplementedError
