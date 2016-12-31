import time
import numpy as np
from collections import OrderedDict
import ogre
import spaces
from envs import OgreEnv, StraightCarOgreEnv
import utils
import utils.transformations as tf


class SimpleQuadOgreEnv(OgreEnv):
    def __init__(self, action_space, observation_space, state_space, sensor_names,
                 car_env_class=None, car_action_space=None, car_model_name='camaro2',
                 app=None, dt=None):
        self.car_action_space = car_action_space or spaces.BoxSpace(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        # self.car_action_space = spaces.BoxSpace(np.array([0.0, -0.1]), np.array([1.0, 0.1]))
        # self.car_env = StraightCarOgreEnv(self.car_action_space,
        self.car_env_class = car_env_class or StraightCarOgreEnv
        # self.car_color = car_color or 'red'
        self.car_model_name = car_model_name
        self.car_env = self.car_env_class(self.car_action_space,
                                          None,  # no need to observe from car
                                          None,
                                          app=app,
                                          dt=dt,
                                          model_name=car_model_name)
        app, dt = self.car_env.app, self.car_env.dt

        if isinstance(state_space, spaces.TranslationAxisAngleSpace):
            quad_state_space = state_space
            car_state_space = self.car_env.state_space
            state_space = spaces.TupleSpace([quad_state_space, car_state_space])
        if not (isinstance(state_space, spaces.TupleSpace) and
                    len(state_space.spaces) == 2 and
                    isinstance(state_space.spaces[0], spaces.TranslationAxisAngleSpace)):
            raise ValueError("state_space should be of type TupleSpace with the first space "
                             "being of type TranslationAxisAngleSpace, but instead got "
                             "%r" % state_space)

        super(SimpleQuadOgreEnv, self).__init__(action_space, observation_space, state_space, sensor_names, app=app, dt=dt)

        # modify the car's speed limits so that the car's speed is always a quater of the quad's maximum forward velocity
        self.car_env.state_space.low[0] = self.action_space.high[1] / 4  # meters per second
        self.car_env.state_space.high[0] = self.action_space.high[1] / 4
        self.car_node = self.car_env.car_node
        self.city_node = self.car_env.city_node
        self.skybox_node = self.car_env.skybox_node

        with utils.suppress_stdout():
            quad_entity = self.app.scene_manager.createEntity('iris_obj.mesh')
        self.quad_node = self.app.scene_manager.getRootSceneNode().createChildSceneNode('quad')
        quad_local_node = self.quad_node.createChildSceneNode()
        quad_local_node.attachObject(quad_entity)
        quad_local_node.setOrientation(tf.quaternion_about_axis(-np.pi / 2, np.array([0, 0, 1])))

        quad_prop_positions = [np.array([ 0.20610,  0.1383 , 0.02]),  # blue, right
                               np.array([ 0.22254, -0.12507, 0.02]),  # black, right
                               np.array([-0.20266,  0.1383 , 0.02]),  # blue, left
                               np.array([-0.21911, -0.12507, 0.02])]  # black, left
        self.quad_prop_local_nodes = []
        for quad_prop_id, quad_prop_pos in enumerate(quad_prop_positions):
            with utils.suppress_stdout():
                is_ccw = quad_prop_id in (1, 2)
                quad_prop_entity = self.app.scene_manager.createEntity('iris_prop_%s_dae.mesh' % ('ccw' if is_ccw else 'cw'))
            quad_prop_node = self.quad_node.createChildSceneNode('quad_prop_%d' % quad_prop_id)
            quad_prop_node.setPosition(quad_prop_pos)
            quad_prop_node.setOrientation(tf.quaternion_about_axis(np.pi / 2, np.array([1, 0, 0])))
            quad_prop_local_node = quad_prop_node.createChildSceneNode()
            quad_prop_local_node.attachObject(quad_prop_entity)
            self.quad_prop_local_nodes.append(quad_prop_local_node)
        self.prop_angle = 0.0
        self.prop_rpm = 10212

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
        if self.prop_rpm:
            self.prop_angle += (self.prop_rpm * 2 * np.pi / 60) * self.dt
            self.prop_angle -= 2 * np.pi * np.floor(self.prop_angle / (2 * np.pi))
            for quad_prop_id, quad_prop_local_node in enumerate(self.quad_prop_local_nodes):
                is_ccw = quad_prop_id in (1, 2)
                angle = self.prop_angle if is_ccw else -self.prop_angle
                quad_prop_local_node.setOrientation(tf.quaternion_about_axis(angle, np.array([0, 1, 0])))

        car_action = self.car_env.action_space.sample()
        self.car_env.step(car_action)

        # import IPython as ipy; ipy.embed()
        # for i in range(10):
        #     car_action = self.car_env.action_space.sample()
        #     self.car_env.step(car_action)
        #     print(self.car_env.max_straight_dist, self.car_env.turn_dist_offset, self.car_env.max_turn_angle, self.car_env.start_turn_angle_offset, self.car_env.end_turn_angle_offset)
        #     self.render()s
        #
        # action = np.array([0.0, 10.0, 0.0, 0.0])

        # update action to be within the action space
        if not self.action_space.contains(action):
            action = self.action_space.clip(action, out=action)

        linear_vel, angular_vel = np.split(action, [3])
        if angular_vel.shape == (1,):
            angular_vel = angular_vel * self._action_space.axis

        # compute next state
        quad_T = self.quad_node.getTransform()
        quad_to_next_quad_T = tf.position_axis_angle_matrix(np.append(linear_vel, angular_vel) * self.dt)
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

        linear_vel, angular_vel = np.split(tf.position_axis_angle_from_matrix(quad_to_next_quad_T) / self.dt, [3])
        # project angular_vel onto the axis
        if self._action_space.axis is not None:
            angular_vel = angular_vel.dot(self._action_space.axis)
        action[:] = np.append(linear_vel, angular_vel)

    def get_state(self):
        quad_T = self.quad_node.getTransform()
        quad_state = tf.position_axis_angle_from_matrix(quad_T)
        car_state = self.car_env.get_state()
        return np.concatenate([quad_state, car_state])

    def reset(self, state=None):
        self._first_render = True
        if state is None:
            quad_state, _ = self.state_space.sample()
            self.car_env.reset(state=None)
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
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -4., 3.]) * 1.2)
        self.app.camera.setPosition((1 - tightness) * self.app.camera.getPosition() + tightness * target_camera_pos)
        if self.app.window.isHidden():
            self.app.window.setHidden(False)
        self.app.root.renderOneFrame()
        self.app.window.update()

    @property
    def state(self):
        return tf.position_axis_angle_from_pose(self.quad_node.getOrientation(), self.quad_node.getPosition())

    def _get_config(self):
        config = super(SimpleQuadOgreEnv, self)._get_config()
        config.update({'car_env_class': self.car_env_class,
                       'car_action_space': self.car_action_space,
                       'car_model_name': self.car_model_name})
        return config


def main():
    action_space = spaces.TranslationAxisAngleSpace(np.array([-2, -2, -2, -np.deg2rad(10)]),
                                              np.array([2, 2, 2, np.deg2rad(10)]))
    observation_space = spaces.TupleSpace([spaces.BoxSpace(0, 255, shape=(480, 640), dtype=np.uint8),
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
