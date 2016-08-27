import time
import numpy as np
import ogre
import spaces
import policy
from envs import OgreEnv
import utils.transformations as tf


class ObjectOgreEnv(OgreEnv):
    def __init__(self, action_space, observation_space, state_space, sensor_names, object_fname, app=None, dt=None):
        super(ObjectOgreEnv, self).__init__(action_space, observation_space, state_space, sensor_names, app=app, dt=dt)
        self.object_fname = object_fname

        object_entity = self.app.scene_manager.createEntity(object_fname)
        self.object_node = self.app.scene_manager.getRootSceneNode().createChildSceneNode()
        self.object_node.attachObject(object_entity)

        self.camera_node = self.app.scene_manager.getRootSceneNode().createChildSceneNode()
        camera = self.app.scene_manager.createCamera('camera')
        camera.setNearClipDistance(self.app.camera.getNearClipDistance())
        camera.setFarClipDistance(self.app.camera.getFarClipDistance())
        self.camera_node.attachObject(camera)
        self.camera_node.setOrientation(np.array([1., 0., 0., 0.]))
        self.camera_node.setPosition(np.array([0., 0., 50.]))

        if 'image' in self.sensor_names:
            self.camera_sensor = ogre.PyCameraSensor(camera, 640, 480)
        if 'depth_image' in self.sensor_names:
            self.depth_camera_sensor = ogre.PyDepthCameraSensor(camera, 640, 480)

    def step(self, action):
        # update action to be within the action space
        if not self.action_space.contains(action):
            action = self.action_space.clip(action, out=action)

        # compute next state
        object_T = self.object_node.getTransform()
        object_to_next_object_T = tf.position_axis_angle_matrix(action * self.dt)
        next_object_T = object_T @ object_to_next_object_T

        # set new state
        self.object_node.setTransform(next_object_T)

    def get_state(self):
        object_T = self.object_node.getTransform()
        return tf.axis_angle_from_matrix(object_T)

    def reset(self, state=None):
        if state is None:
            state = self.state_space.sample()
        object_T = tf.position_axis_angle_matrix(state)
        self.object_node.setTransform(object_T)

    def observe(self):
        obs = []
        for sensor_name in self.sensor_names:
            if sensor_name == 'image':
                observation = self.camera_sensor.observe()
            elif sensor_name == 'depth_image':
                observation = self.depth_camera_sensor.observe()
            else:
                raise ValueError('Unknown sensor name %s' % sensor_name)
            obs.append(observation.copy())
        return tuple(obs)

    def render(self):
        self.app.camera.setOrientation(np.array([1., 0., 0., 0.]))
        self.app.camera.setPosition(np.array([0., 0., 50.]))
        self.app.root.renderOneFrame()
        self.app.window.update()

    def _get_config(self):
        config = super(ObjectOgreEnv, self)._get_config()
        config.update({'object_fname': self.object_fname})
        return config


def main():
    action_space = spaces.TranslationAxisAngleSpace(np.array([-2, -2, -2, -np.pi / 4]),
                                                    np.array([2, 2, 2, np.pi / 4]))
    observation_space = spaces.Tuple([spaces.BoxSpace(0, 255, shape=(480, 640), dtype=np.uint8),
                                      spaces.BoxSpace(0.0, np.inf, shape=(480, 640))])
    state_space = spaces.TranslationAxisAngleSpace(np.array([0, 0, 0, -np.inf]), np.array([0, 0, 0, np.inf]))
    sensor_names = ['image', 'depth_image']
    object_fname = 'camaro2_3ds.mesh'
    object_env = ObjectOgreEnv(action_space, observation_space, state_space, sensor_names, object_fname)

    pol = policy.ConstantPolicy(action_space, state_space)

    for i in range(10):
        state = pol.reset()
        object_env.reset(state)
        for t in range(20):
            start_time = time.time()
            obs = object_env.observe()
            action = pol.act(obs)
            object_env.step(action)
            object_env.render()
            time.sleep(max(object_env.dt + start_time - time.time(), 0))


if __name__ == "__main__":
    main()
