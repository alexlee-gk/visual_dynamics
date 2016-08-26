import numpy as np
from policy import Policy


class RandomPolicy(Policy):
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space

    def act(self, obs):
        return self.action_space.sample()

    def reset(self):
        state = self.state_space.sample()
        if isinstance(state, tuple):
            state = np.concatenate(state)
        return state

    def _get_config(self):
        config = super(RandomPolicy, self)._get_config()
        config.update({'action_space': self.action_space,
                       'state_space': self.state_space})
        return config
