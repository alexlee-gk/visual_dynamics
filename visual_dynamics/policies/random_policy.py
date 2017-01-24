import numpy as np

from visual_dynamics.policies import TargetPolicy


class RandomPolicy(TargetPolicy):
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self._state = None

    def act(self, obs):
        return self.action_space.sample()

    def reset(self):
        state = self.state_space.sample()
        if isinstance(state, tuple):
            state = np.concatenate(state)
        self._state = state
        return state

    def get_target_state(self):
        return self._state

    def _get_config(self):
        config = super(RandomPolicy, self)._get_config()
        config.update({'action_space': self.action_space,
                       'state_space': self.state_space})
        return config
