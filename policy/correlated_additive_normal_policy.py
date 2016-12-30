import numpy as np
import scipy.stats
from policy import Policy
import spaces


class CorrelatedAdditiveNormalPolicy(Policy):
    def __init__(self, pol, action_space, state_space, act_std=None, reset_std=None, averaging_window=4):
        self.pol = pol
        self.action_space = action_space
        self.state_space = state_space
        self.act_std = act_std
        self.reset_std = reset_std
        self.averaging_window = averaging_window
        self._sampled_actions = []

    @staticmethod
    def truncated_normal(mean, std, space):
        if std is None:
            return mean
        std = std * (space.high - space.low) / 2.
        tn = scipy.stats.truncnorm((space.low - mean) / std, (space.high - mean) / std, mean, std)
        if isinstance(space, (spaces.AxisAngleSpace, spaces.TranslationAxisAngleSpace)) and \
                space.axis is None:
            raise NotImplementedError
        return tn.rvs()

    def act(self, obs):
        action = self.truncated_normal(self.pol.act(obs), self.act_std, self.action_space)
        self._sampled_actions.append(action)
        del self._sampled_actions[:-self.averaging_window]
        return np.mean(self._sampled_actions, axis=0)

    def reset(self):
        return self.truncated_normal(self.pol.reset(), self.reset_std, self.state_space)

    def _get_config(self):
        config = super(CorrelatedAdditiveNormalPolicy, self)._get_config()
        config.update({'pol': self.pol,
                       'action_space': self.action_space,
                       'state_space': self.state_space,
                       'act_std': self.act_std,
                       'reset_std': self.reset_std,
                       'averaging_window': self.averaging_window})
        return config
