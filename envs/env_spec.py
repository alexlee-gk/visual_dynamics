from utils import ConfigObject


class EnvSpec(ConfigObject):
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _get_config(self):
        config = super(EnvSpec, self)._get_config()
        config.update({'action_space': self.action_space,
                       'observation_space': self.observation_space})
        return config
