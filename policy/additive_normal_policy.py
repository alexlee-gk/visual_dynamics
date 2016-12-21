import scipy.stats
from policy import Policy
import spaces


class AdditiveNormalPolicy(Policy):
    def __init__(self, pol, action_space, state_space, act_std=None, reset_std=None):
        self.pol = pol
        self.action_space = action_space
        self.state_space = state_space
        self.act_std = act_std
        self.reset_std = reset_std

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
        return self.truncated_normal(self.pol.act(obs), self.act_std, self.action_space)

    def reset(self):
        return self.truncated_normal(self.pol.reset(), self.reset_std, self.state_space)

    def _get_config(self):
        config = super(AdditiveNormalPolicy, self)._get_config()
        config.update({'pol': self.pol,
                       'action_space': self.action_space,
                       'state_space': self.state_space,
                       'act_std': self.act_std,
                       'reset_std': self.reset_std})
        return config
