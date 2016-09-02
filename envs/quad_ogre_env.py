import time
import numpy as np
from collections import OrderedDict
import ogre
import spaces
from envs import OgreEnv, StraightCarOgreEnv
import utils
import utils.transformations as tf


class SimpleQuadOgreEnv(OgreEnv):
    def __init__(self, action_space, observation_space, state_space, sensor_names, car_env_class=None, car_action_space=None, app=None, dt=None):
        self.car_action_space = car_action_space or spaces.BoxSpace(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        # self.car_action_space = spaces.BoxSpace(np.array([0.0, -0.1]), np.array([1.0, 0.1]))
        # self.car_env = StraightCarOgreEnv(self.car_action_space,
        self.car_env_class = car_env_class or StraightCarOgreEnv
        self.car_env = self.car_env_class(self.car_action_space,
                                          None,  # no need to observe from car
                                          None,
                                          app=app,
                                          dt=dt)
        app, dt = self.car_env.app, self.car_env.dt

        if isinstance(state_space, spaces.TranslationAxisAngleSpace):
            quad_state_space = state_space
            car_state_space = self.car_env.state_space
            state_space = spaces.Tuple([quad_state_space, car_state_space])
        if not (isinstance(state_space, spaces.Tuple) and
                    len(state_space.spaces) == 2 and
                    isinstance(state_space.spaces[0], spaces.TranslationAxisAngleSpace)):
            raise ValueError("state_space should be of type Tuple with the first space "
                             "being of type TranslationAxisAngleSpace, but instead got "
                             "%r" % state_space)

        super(SimpleQuadOgreEnv, self).__init__(action_space, observation_space, state_space, sensor_names, app=app, dt=dt)

        # modify the car's speed limits so that the car's speed is always a quater of the quad's maximum forward velocity
        self.car_env.state_space.low[0] = self.action_space.high[1] / 4  # meters per second
        self.car_env.state_space.high[0] = self.action_space.high[1] / 4
        self.car_node = self.car_env.car_node
        self.city_node = self.car_env.city_node
        self.skybox_node = self.car_env.skybox_node

        quad_entity = self.app.scene_manager.createEntity('iris_dae.mesh')
        self.quad_node = self.app.scene_manager.getRootSceneNode().createChildSceneNode('quad')
        quad_local_node = self.quad_node.createChildSceneNode()
        quad_local_node.attachObject(quad_entity)
        quad_local_node.setOrientation(
            tf.quaternion_multiply(tf.quaternion_about_axis(np.pi / 2, np.array([1, 0, 0])),
                                   tf.quaternion_about_axis(np.pi / 2, np.array([0, 0, 1]))))

        self.quad_camera_node = self.quad_node.createChildSceneNode('quad_camera')
        quad_camera = self.app.scene_manager.createCamera('quad_camera')
        quad_camera.setNearClipDistance(self.app.camera.getNearClipDistance())
        quad_camera.setFarClipDistance(self.app.camera.getFarClipDistance())
        self.quad_camera_node.attachObject(quad_camera)
        self.quad_camera_node.setPosition(np.array([0, -4., 3.]) * -0.02)  # slightly in front of the quad
        self.quad_camera_node.setOrientation(tf.quaternion_about_axis(np.pi / 3, np.array([1, 0, 0])))

        if 'image' in self.sensor_names:
            self.quad_camera_sensor = ogre.PyCameraSensor(quad_camera, 640, 480)
        if 'depth_image' in self.sensor_names:
            self.quad_depth_camera_sensor = ogre.PyDepthCameraSensor(quad_camera, 640, 480)

        self._first_render = True
        self._start_time = time.time()

    def step(self, action):
        car_action = self.car_env.action_space.sample()
        self.car_env.step(car_action)

        # update action to be within the action space
        if not self.action_space.contains(action):
            action = self.action_space.clip(action, out=action)

        # compute next state
        quad_T = self.quad_node.getTransform()
        quad_to_next_quad_T = tf.position_axis_angle_matrix(action * self.dt)
        # TODO: better handling of 3dof action
        # if action.shape == (3,):
        #     assert np.allclose(quad_to_next_quad_T[:3, :3], np.eye(3))
        #     quad_to_next_quad_T[:3, :3] = np.eye(3)
        next_quad_T = quad_T.dot(quad_to_next_quad_T)

        # update the state to be within the state space
        next_quad_pos_aa = tf.position_axis_angle_from_matrix(next_quad_T)
        if not self.state_space.spaces[0].contains(next_quad_pos_aa):
            next_quad_pos_aa = self.state_space.spaces[0].clip(next_quad_pos_aa, out=next_quad_pos_aa)
            next_quad_T = tf.position_axis_angle_matrix(next_quad_pos_aa)

        # set new state
        self.quad_node.setTransform(next_quad_T)

        # update action to be consistent with the state clippings
        quad_to_next_quad_T = tf.inverse_matrix(quad_T).dot(next_quad_T)
        # TODO: better handling of 3dof action
        # if action.shape == (3,):
        #     assert np.allclose(quad_to_next_quad_T[:3, :3], np.eye(3))
        #     action[:] = tf.position_axis_angle_from_matrix(quad_to_next_quad_T)[:3] / self.dt
        # else:
        #     action[:] = tf.position_axis_angle_from_matrix(quad_to_next_quad_T) / self.dt
        action[:], action_rem = np.split(tf.position_axis_angle_from_matrix(quad_to_next_quad_T) / self.dt,
                                         action.shape)
        assert all(action_rem == 0)  # action_rem may be empty, in which case this is also True

    def get_state(self):
        quad_T = self.quad_node.getTransform()
        quad_state = tf.position_axis_angle_from_matrix(quad_T)
        car_state = self.car_env.get_state()
        return np.concatenate([quad_state, car_state])

    def reset(self, state=None):
        self._first_render = True
        if state is None:
            quad_state, car_state = self.state_space.sample()
        else:
            quad_state, car_state = np.split(state, [6])
        self.car_env.reset(car_state)
        # TODO: verify that the car speed doesn't need to be set every time
        # self.car_env.speed = self.action_space.high[1] / 4
        quad_T = tf.position_axis_angle_matrix(quad_state)
        self.quad_node.setTransform(quad_T)

    def get_error_names(self):
        return ['position', 'rotation']

    def get_errors(self, target_state):
        target_T = tf.position_axis_angle_matrix(target_state[:6])
        quad_T = tf.position_axis_angle_matrix(self.get_state()[:6])
        quad_to_target_T = tf.inverse_matrix(quad_T).dot(target_T)
        pos_error = np.linalg.norm(quad_to_target_T[:3, 3])
        angle_error = np.linalg.norm(tf.axis_angle_from_matrix(quad_to_target_T))
        return OrderedDict([('position', pos_error), ('rotation', angle_error)])

    # def reset(self, state_or_policy=None):
    #     # allow to pass in policy in case that the reset state of the policy depends on the state of the car
    #     self._first_render = True
    #     if state_or_policy is None:
    #         quad_state, car_state = self.state_space.sample()
    #     else:
    #         car_state = self.car_env.state_space.sample()
    #     self.car_env.reset(car_state)
    #     # TODO: verify that the car speed doesn't need to be set every time
    #     # self.car_env.speed = self.action_space.high[1] / 4
    #     if state_or_policy is not None:
    #         if isinstance(state_or_policy, policy.Policy):
    #             quad_state = state_or_policy.reset()
    #         else:
    #             quad_state = state_or_policy
    #     quad_T = tf.position_axis_angle_matrix(quad_state)
    #     self.quad_node.setTransform(quad_T)

    def observe(self):
        obs = []
        for sensor_name in self.sensor_names:
            if sensor_name == 'image':
                observation = self.quad_camera_sensor.observe()
            elif sensor_name == 'depth_image':
                observation = self.quad_depth_camera_sensor.observe()
            elif sensor_name == 'car_image':
                self.car_env.city_env.city_node.setVisible(False)
                self.car_env.city_env.skybox_node.setVisible(False)
                observation = self.quad_camera_sensor.observe()
                self.car_env.city_env.city_node.setVisible(True)
                self.car_env.city_env.skybox_node.setVisible(True)
            elif sensor_name == 'car_depth_image':
                self.car_env.city_env.city_node.setVisible(False)
                self.car_env.city_env.skybox_node.setVisible(False)
                observation = self.quad_depth_camera_sensor.observe()
                self.car_env.city_env.city_node.setVisible(True)
                self.car_env.city_env.skybox_node.setVisible(True)
            else:
                raise ValueError('Unknown sensor name %s' % sensor_name)
            obs.append(observation.copy())
        return tuple(obs)

    def render(self):
        target_node = self.quad_node
        if self._first_render:
            self.app.camera.setAutoTracking(True, target_node)
            self.app.camera.setFixedYawAxis(True, np.array([0., 0., 1.]))
            tightness = 1.0
            self._first_render = False
        else:
            tightness = 0.1
        target_T = tf.pose_matrix(target_node._getDerivedOrientation(), target_node._getDerivedPosition())
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -4., 3.]) * 2)
        self.app.camera.setPosition((1 - tightness) * self.app.camera.getPosition() + tightness * target_camera_pos)
        self.app.root.renderOneFrame()
        self.app.window.update()

    @property
    def state(self):
        return tf.position_axis_angle_from_pose(self.quad_node.getOrientation(), self.quad_node.getPosition())

    def _get_config(self):
        config = super(SimpleQuadOgreEnv, self)._get_config()
        config.update({'car_env_class': self.car_env_class,
                       'car_action_space': self.car_action_space})
        return config


def main():
    action_space = spaces.TranslationAxisAngleSpace(np.array([-2, -2, -2, -np.deg2rad(10)]),
                                              np.array([2, 2, 2, np.deg2rad(10)]))
    observation_space = spaces.Tuple([spaces.BoxSpace(0, 255, shape=(480, 640), dtype=np.uint8),
                                      spaces.BoxSpace(0.0, np.inf, shape=(480, 640))])
    sensor_names = ['image', 'depth_image', 'car_image', 'car_depth_image']
    state_space = spaces.TranslationAxisAngleSpace(np.array([-np.inf] * 4),
                                             np.array([np.inf] * 4))
    quad_env = SimpleQuadOgreEnv(action_space, observation_space, sensor_names, state_space)

    quad_action = np.zeros(6)
    while True:
        quad_env.step(quad_action)
        obs = quad_env.observe()
        done, key = utils.visualization.visualize_images_callback(*obs)
        if done:
            break
        quad_action = np.zeros(6)
        if key == 81:  # left arrow
            quad_action[0] = -2
        elif key == 82:  # up arrow
            quad_action[1] = 2
        elif key == 83:  # right arrow
            quad_action[0] = 2
        elif key == 84:  # down arrow
            quad_action[1] = -2
        elif key == ord('w'):
            quad_action[2] = 2
        elif key == ord('s'):
            quad_action[2] = -2
        elif key == ord('a'):
            quad_action[3:] = np.deg2rad(10.0) * np.array([0, 0, 1.])
        elif key == ord('d'):
            quad_action[3:] = -np.deg2rad(10.0) * np.array([0, 0, 1.])
        elif key == 32:  # space
            car_T = tf.pose_matrix(quad_env.car_env.car_node.getOrientation(), quad_env.car_env.car_node.getPosition())
            quad_T = car_T.dot(tf.translation_matrix(np.array([0., -4., 3.]) * 4))
            quad_env.quad_node.setTransform(quad_T)

if __name__ == "__main__":
    main()
