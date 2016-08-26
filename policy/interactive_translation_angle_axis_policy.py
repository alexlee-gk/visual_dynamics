import numpy as np
import matplotlib
matplotlib.rcParams['keymap.save'].remove('s')
from policy import Policy
import spaces


class InteractiveTranslationAngleAxisPolicy(Policy):
    def __init__(self, fig, action_space):
        assert isinstance(action_space, spaces.TranslationAxisAngleSpace)
        self._fig = fig
        self.action_space = action_space
        self._pressed_keys = set()
        self._cid = self._fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._keyboard_bindings = {'left':  (0, self.action_space.low[0]),
                                   'right': (0, self.action_space.high[0]),
                                   'down':  (1, self.action_space.low[1]),
                                   'up':    (1, self.action_space.high[1]),
                                   's':     (2, self.action_space.low[2]),
                                   'w':     (2, self.action_space.high[2]),
                                   'd':     (5, self.action_space.low[3]),
                                   'a':     (5, self.action_space.high[3])}

    def act(self, obs):
        action = np.zeros(6)
        for key in self._pressed_keys:
            if key in self._keyboard_bindings:
                ind, val = self._keyboard_bindings[key]
                action[ind] = val
        self._pressed_keys.clear()
        return action

    def _on_key_press(self, event):
        self._pressed_keys.add(event.key)

    def _get_config(self):
        config = super(InteractiveTranslationAngleAxisPolicy, self)._get_config()
        config.update({'action_space': self.action_space})
        return config
