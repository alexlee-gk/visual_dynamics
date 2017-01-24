import os

import citysim3d.envs
from panda3d.core import loadPrcFile

from visual_dynamics.envs import Env


assert "CITYSIM3D_DIR" in os.environ
loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))


class Panda3dEnv(citysim3d.envs.Panda3dEnv, Env):
    def _get_config(self):
        config = super(Panda3dEnv, self)._get_config()
        config.update({'dt': self.dt})
        return config
