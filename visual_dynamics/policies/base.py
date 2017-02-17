from visual_dynamics.utils.config import ConfigObject


class Policy(ConfigObject):
    def act(self, obs):
        """
        Returns the action of this policy given the observation, and this
        action may be non-deterministic.

        Args:
            obs: observation

        Returns:
            the action of this policy
        """
        raise NotImplementedError

    def reset(self):
        """
        Returns:
            the reset state of this policies
        """
        raise NotImplementedError
