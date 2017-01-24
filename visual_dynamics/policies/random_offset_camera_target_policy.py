import numpy as np

from visual_dynamics.policies import CameraTargetPolicy


class RandomOffsetCameraTargetPolicy(CameraTargetPolicy):
    def __init__(self, env, target_env, camera_node_name, agent_node_name, target_node_name,
                 height=12.0, radius=16.0, angle=(-np.pi/4, np.pi/4), tightness=0.1, hra_interpolation=True):
        self.height = height
        self.radius = radius
        self.angle = angle
        offset = self.sample_offset()
        super(RandomOffsetCameraTargetPolicy, self).__init__(env, target_env, camera_node_name, agent_node_name,
                                                             target_node_name, offset, tightness=tightness,
                                                             hra_interpolation=hra_interpolation)

    def reset(self):
        self.offset = self.sample_offset()
        state = super(RandomOffsetCameraTargetPolicy, self).reset()
        # self.offset = self.sample_offset()
        return state

    def sample_offset(self):
        height = np.random.uniform(*self.height) if isinstance(self.height, (list, tuple)) else self.height
        radius = np.random.uniform(*self.radius) if isinstance(self.radius, (list, tuple)) else self.radius
        angle = np.random.uniform(*self.angle) if isinstance(self.angle, (list, tuple)) else self.angle
        return np.array([radius * np.sin(angle), -radius * np.cos(angle), height])

    def _get_config(self):
        config = super(RandomOffsetCameraTargetPolicy, self)._get_config()
        config.pop('offset')
        config.update({'height': self.height,
                       'radius': self.radius,
                       'angle': self.angle})
        return config
