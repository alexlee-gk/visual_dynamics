import citysim3d.envs

from envs import Env


class ServoingEnv(citysim3d.envs.ServoingEnv, Env):
    def _get_config(self):
        config = super(ServoingEnv, self)._get_config()
        config.update({'env': self.env,
                       'max_time_steps': self.max_time_steps,
                       'distance_threshold': self.distance_threshold})
        return config
