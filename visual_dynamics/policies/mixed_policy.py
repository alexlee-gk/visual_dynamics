import numpy as np

from visual_dynamics.policies import Policy


class MixedPolicy(Policy):
    def __init__(self, policies, probs=None, act_probs=None, reset_probs=None):
        """
        Args:
            policies: the policies to mix
            probs: the probabilities of each policy. If not given, a uniform
                distribution is used.
        """
        self.policies = policies
        self.act_probs = act_probs if act_probs is not None else probs
        self.reset_probs = reset_probs if reset_probs is not None else probs

    def act(self, obs):
        ind = np.random.choice(len(self.policies), p=self.act_probs)
        return self.policies[ind].act(obs)

    def reset(self):
        ind = np.random.choice(len(self.policies), p=self.reset_probs)
        return self.policies[ind].reset()

    def _get_config(self):
        config = super(MixedPolicy, self)._get_config()
        config.update({'policies': self.policies,
                       'act_probs': self.act_probs,
                       'reset_probs': self.reset_probs})
        return config
