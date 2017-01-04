import citysim3d.envs
from envs import Panda3dEnv
from spaces import Space
import utils


class SimpleQuadPanda3dEnv(citysim3d.envs.SimpleQuadPanda3dEnv, Panda3dEnv):
    def _get_config(self):
        config = super(SimpleQuadPanda3dEnv, self)._get_config()
        car_action_space = self.car_action_space
        if not isinstance(car_action_space, utils.ConfigObject):
            car_action_space = Space.create(car_action_space)
        config.update({'action_space': self.action_space,
                       'sensor_names': self.sensor_names,
                       'offset': self.offset.tolist(),
                       'car_env_class': self.car_env_class,
                       'car_action_space': car_action_space,
                       'car_model_names': self.car_model_names})
        return config


def main():
    import os
    import numpy as np
    from panda3d.core import loadPrcFile
    import spaces

    assert "CITYSIM3D_DIR" in os.environ
    loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))

    action_space = spaces.TranslationAxisAngleSpace(np.array([-20, -10, -10, -1.5707963267948966]),
                                                    np.array([20, 10, 10, 1.5707963267948966]))
    sensor_names = ['image', 'depth_image']
    env = SimpleQuadPanda3dEnv(action_space, sensor_names)

    import time
    import cv2
    start_time = time.time()
    frames = 0

    from policy.quad_target_policy import QuadTargetPolicy
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
