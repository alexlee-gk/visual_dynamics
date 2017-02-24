import numpy as np
import citysim3d.envs

from visual_dynamics.envs import Panda3dEnv
from visual_dynamics.spaces import Space, BoxSpace, TranslationAxisAngleSpace
from visual_dynamics.utils.config import ConfigObject


class SimpleQuadPanda3dEnv(citysim3d.envs.SimpleQuadPanda3dEnv, Panda3dEnv):
    def _get_config(self):
        config = super(SimpleQuadPanda3dEnv, self)._get_config()
        car_action_space = self.car_action_space
        if not isinstance(car_action_space, ConfigObject):
            car_action_space = Space.create(car_action_space)
        config.update({'action_space': self.action_space,
                       'sensor_names': self.sensor_names,
                       'camera_size': self.camera_size,
                       'camera_hfov': self.camera_hfov,
                       'offset': self.offset.tolist(),
                       'car_env_class': self.car_env_class,
                       'car_action_space': car_action_space,
                       'car_model_names': self.car_model_names})
        return config


class Point3dSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    def __init__(self, action_space, **kwargs):
        super(Point3dSimpleQuadPanda3dEnv, self).__init__(action_space, **kwargs)
        self._observation_space.spaces['pos'] = BoxSpace(-np.inf, np.inf, shape=(3,))

    def observe(self):
        obs = super(Point3dSimpleQuadPanda3dEnv, self).observe()
        obs['pos'] = np.array(self.car_node.getTransform(self.camera_node).getPos())
        return obs


def main():
    import os
    import numpy as np
    from panda3d.core import loadPrcFile

    assert "CITYSIM3D_DIR" in os.environ
    loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))

    action_space = TranslationAxisAngleSpace(np.array([-20, -10, -10, -1.5707963267948966]),
                                             np.array([20, 10, 10, 1.5707963267948966]))
    sensor_names = ['image', 'depth_image']
    env = SimpleQuadPanda3dEnv(action_space, sensor_names)

    import time
    import cv2
    start_time = time.time()
    frames = 0

    from visual_dynamics.policies.quad_target_policy import QuadTargetPolicy
    pol = QuadTargetPolicy(env, (12, 18), (-np.pi / 2, np.pi / 2))

    obs = env.reset()
    pol.reset()
    image, depth_image = obs
    while True:
        try:
            env.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Image window', image)
            key = cv2.waitKey(1)
            key &= 255
            if key == 27 or key == ord('q'):
                print("Pressed ESC or q, exiting")
                break

            quad_action = pol.act(obs)
            obs, _, _, _ = env.step(quad_action)
            image, depth_image = obs
            frames += 1
        except KeyboardInterrupt:
            break

    end_time = time.time()
    print("average FPS: {}".format(frames / (end_time - start_time)))


if __name__ == "__main__":
    main()
