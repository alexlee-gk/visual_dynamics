import numpy as np
from policy import Policy


class ConstantPolicy(Policy):
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self._action = None

    def act(self, obs):
        if self._action is None:
            self._action = self.action_space.sample()
        return self._action

    def reset(self):
        self._action = None
        state = self.state_space.sample()
        if isinstance(state, tuple):
            state = np.concatenate(state)
        return state

    def _get_config(self):
        config = super(ConstantPolicy, self)._get_config()
        config.update({'action_space': self.action_space,
                       'state_space': self.state_space})
        return config
