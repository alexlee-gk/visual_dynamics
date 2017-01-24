from visual_dynamics.utils.config import ConfigObject


class Env(ConfigObject):
    def step(self, action):
        """
        Run one time step of the environment's dynamics

        Args:
            action: numpy array, which should be contained in the action space
                or be clippable by the action space.

        Returns:
            observation (object): agent's observation of the current environment.
            reward (float) : amount of reward returned after previous action.
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        Note:
            The action is modified in-place with the action that was actually
            taken. For example, this could happen if the action is not contained
            in the action space or if applying it leads to an invalid state.
        """
        raise NotImplementedError

    def reset(self, state=None):
        """
        Resets the state of the environment and returns an initial observation

        Args:
            state: numpy array, which is the state this environment should be
                set to (if specified)

        If state is specified, the environment is reset to that state,
        otherwise, it is set to an arbitrary state (which may be chosen at
        random).

        Returns:
            observation: the initial observation of the environment.
        """
        raise NotImplementedError

    def get_state(self):
        """
        Returns the state of the environment

        Returns:
            a numpy array.
        """
        raise NotImplementedError

    def set_state(self, state):
        """
        Sets the state of the environment

        Args:
            state: numpy array.

        Note:
            Setting the state of the environment to the current state should
            not affect the state of the environment.

        Example:

            >>> state = self.get_state()
            >>> self.set_state(state)
            >>> assert np.allclose(state, self.get_state())

        """
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        raise NotImplementedError
