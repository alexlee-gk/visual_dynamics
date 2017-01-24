from visual_dynamics.spaces.base import Space
from visual_dynamics.utils.config import ConfigObject


class EnvSpec(ConfigObject):
    def __init__(self, action_space, observation_space):
        self._action_space = action_space
        self._observation_space = observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _get_config(self):
        config = super(EnvSpec, self)._get_config()
        action_space = self.action_space
        if not isinstance(action_space, ConfigObject):
            action_space = Space.create(action_space)
        observation_space = self.observation_space
        if not isinstance(observation_space, ConfigObject):
            observation_space = Space.create(observation_space)
        config.update({'action_space': action_space,
                       'observation_space': observation_space})
        return config
