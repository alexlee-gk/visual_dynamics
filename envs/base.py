import utils


class Env(utils.config.ConfigObject):
    def __init__(self, action_space, observation_space, state_space, sensor_names):
        self._action_space = action_space
        self._observation_space = observation_space
        self._state_space = state_space
        self._sensor_names = sensor_names or []

    def step(self, action):
        """
        The action should be contained in the action_space.
        """
        raise NotImplementedError

    def get_state(self):
        """
        The returned state should be contained in the state_space.
        """
        raise NotImplementedError

    def reset(self, state=None):
        """
        The state (if specified) should be contained in the state_space.
        """
        raise NotImplementedError

    def observe(self):
        """
        Returns a tuple of observations even if there is only one observation.

        Returns:
            a tuple of observations, where each observations is a numpy array,
            and the number of observations should match the number of
            sensor_names. The observations should be contained in the
            observation space.
        """
        raise NotImplementedError

    def get_state_and_observe(self):
        """
        Returns the state of get_state() and the observations of observe(),
        ensuring that they correspond to each other. This is particularly
        useful in real-world systems.

        Returns:
            a tuple of the state and the observations

        """
        return self.get_state(), self.observe()

    def render(self):
        pass

    def close(self):
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def sensor_names(self):
        return self._sensor_names

    def _get_config(self):
        config = super(Env, self)._get_config()
        config.update({'action_space': self.action_space,
                       'observation_space': self.observation_space,
                       'state_space': self.state_space,
                       'sensor_names': self.sensor_names})
        return config
