import numpy as np
from envs import Env
import ogre


class OgreEnv(Env):
    def __init__(self, action_space, observation_space, state_space, sensor_names, app=None, dt=None):
        super(OgreEnv, self).__init__(action_space, observation_space, state_space, sensor_names)
        if app is None:
            self.app = ogre.PyApplication()
            if not self.app.setup():
                raise Exception("Failed to setup ogre")
        else:
            self.app = app
        self.app.camera.setNearClipDistance(0.01)  # 1cm
        self.app.camera.setFarClipDistance(10000.0)  # 10km
        self._dt = 0.1 if dt is None else 0.1

    def render(self):
        self.app.camera.setOrientation(np.array([1., 0., 0., 0.]))
        self.app.camera.setPosition(np.array([0., 0., 1000.]))
        self.app.root.renderOneFrame()
        self.app.window.update()

    @property
    def dt(self):
        return self._dt

    def _get_config(self):
        config = super(OgreEnv, self)._get_config()
        config.update({'dt': self.dt})
        return config
