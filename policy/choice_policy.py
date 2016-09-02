import numpy as np
from policy import Policy


class ChoicePolicy(Policy):
    def __init__(self, policies, reset_probs=None):
        """
        Unlike a MixedPolicy, a ChoicePolicy randomly picks one of the policies
        when it is reset but then keeps using that policy until a reset happens
        again.

        Args:
            policies: the policies to choose from
        """
        self.policies = policies
        self.reset_probs = reset_probs
        self._ind = None

    def act(self, obs):
        if self._ind is None:
            raise ValueError('reset should be called first before act is called for the first time')
        return self.policies[self._ind].act(obs)

    def reset(self):
        self._ind = np.random.choice(len(self.policies), p=self.reset_probs)
        return self.policies[self._ind].reset()

    def _get_config(self):
        config = super(ChoicePolicy, self)._get_config()
        config.update({'policies': self.policies,
                       'reset_probs': self.reset_probs})
        return config
