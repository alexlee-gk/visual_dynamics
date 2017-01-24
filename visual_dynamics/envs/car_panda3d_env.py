import citysim3d.envs

from visual_dynamics.envs import Panda3dEnv


class CarPanda3dEnv(citysim3d.envs.CarPanda3dEnv, Panda3dEnv):
    def _get_config(self):
        config = super(CarPanda3dEnv, self)._get_config()
        config.update({'action_space': self.action_space,
                       'sensor_names': self.sensor_names,
                       'model_names': self.model_names})
        return config


class StraightCarPanda3dEnv(citysim3d.envs.StraightCarPanda3dEnv, CarPanda3dEnv):
    pass


class SimpleGeometricCarPanda3dEnv(citysim3d.envs.SimpleGeometricCarPanda3dEnv, CarPanda3dEnv):
    pass


class GeometricCarPanda3dEnv(citysim3d.envs.GeometricCarPanda3dEnv, SimpleGeometricCarPanda3dEnv):
    pass


def main():
    import os
    import time
    import numpy as np
    import cv2
    from panda3d.core import loadPrcFile
    from visual_dynamics import spaces

    assert "CITYSIM3D_DIR" in os.environ
    loadPrcFile(os.path.expandvars('${CITYSIM3D_DIR}/config.prc'))

    action_space = spaces.BoxSpace(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    env = GeometricCarPanda3dEnv(action_space)

    start_time = time.time()
    frames = 0

    image = env.reset()
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

            image, _, _, _ = env.step(np.array([0.0, 0.0]))

            frames += 1
        except KeyboardInterrupt:
            break

    end_time = time.time()
    print("average FPS: {}".format(frames / (end_time - start_time)))


if __name__ == '__main__':
    main()
