import citysim3d.envs

from visual_dynamics.envs import Env


class ServoingEnv(citysim3d.envs.SimpleQuadPanda3dServoingEnv, Env):
    def _get_config(self):
        config = super(ServoingEnv, self)._get_config()
        config.update({'env': self.env,
                       'max_time_steps': self.max_time_steps,
                       'distance_threshold': self.distance_threshold})
        return config


# class ServoingEnv(citysim3d.envs.ServoingEnv, Env):
#     def _get_config(self):
#         config = super(ServoingEnv, self)._get_config()
#         config.update({'env': self.env})
#         return config
#
#
# class SimpleQuadPanda3dServoingEnv(citysim3d.envs.SimpleQuadPanda3dServoingEnv, ServoingEnv):
#     def _get_config(self):
#         config = super(SimpleQuadPanda3dServoingEnv, self)._get_config()
#         config.update({'env': self.env,
#                        'max_time_steps': self.max_time_steps,
#                        'distance_threshold': self.distance_threshold})
#         return config
