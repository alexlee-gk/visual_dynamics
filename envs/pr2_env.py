import numpy as np
from collections import OrderedDict
from envs import RosEnv
try:
    import rospy
    from pr2 import PR2, camera_sensor
except:
    pass
import spaces


class Pr2Env(RosEnv):
    def __init__(self, action_space, observation_space, state_space, sensor_names, dt=None):
        super(Pr2Env, self).__init__(action_space, observation_space, state_space, sensor_names)
        self._dt = 0.2 if dt is None else dt
        self.pr2 = PR2.PR2()
        self.pr2.larm.goto_posture('side')
        self.pr2.rarm.goto_posture('side')
        self.pr2.torso.go_down()
        gains = {'head_pan_joint': {'d': 2.0, 'i': 12.0, 'i_clamp': 0.5, 'p': 50.0},
                 'head_tilt_joint': {'d': 3.0, 'i': 4.0, 'i_clamp': 0.2, 'p': 1000.0}}
        rospy.set_param('/head_traj_controller/gains', gains)
        self.pr2.head.set_pan_tilt(*((self.state_space.low + self.state_space.high) / 2.0))

        self.msg_and_camera_sensor = camera_sensor.MessageAndCameraSensor()
        rospy.sleep(5.0)

    def step(self, action):
        # update action to be within the action space
        if not self.action_space.contains(action):
            action = self.action_space.clip(action, out=action)

        pan_tilt_angles = self.pr2.head.get_joint_positions()

        action[:] = self.state_space.clip(pan_tilt_angles + action) - pan_tilt_angles

        self.pr2.head.command_pan_tilt_vel(*action, dt=self._dt)
        rospy.sleep(.3)

    def get_state(self):
        return self.pr2.head.get_joint_positions()

    def reset(self, state=None):
        if state is None:
            state = self.state_space.sample()
        self.pr2.head.goto_joint_positions(state)
        rospy.sleep(1.0)

    def get_error_names(self):
        return ['pan_angle', 'tilt_angle']

    def get_errors(self, target_state):
        pan_error, tilt_error = np.abs(target_state - self.get_state())
        return OrderedDict([('pan_angle', pan_error), ('tilt_angle', tilt_error)])

    def observe(self):
        _, obs = self.get_state_and_observe()
        return obs

    def get_state_and_observe(self):
        joint_state_msg, image = self.msg_and_camera_sensor.get_msg_and_observe()
        state = self.pr2.head.get_joint_positions(msg=joint_state_msg)
        obs = []
        for sensor_name in self.sensor_names:
            if sensor_name == 'image':
                observation = image
            else:
                raise ValueError('Unknown sensor name %s' % sensor_name)
            obs.append(observation.copy())
        return state, obs

    def render(self):
        pass

    @property
    def dt(self):
        return self._dt

    def _get_config(self):
        config = super(Pr2Env, self)._get_config()
        return config


def main():
    rospy.init_node('camera_sensor', anonymous=True)

    action_space = spaces.BoxSpace(np.deg2rad([-5., -5.]), np.deg2rad([5., 5.]))
    observation_space = spaces.TupleSpace([spaces.BoxSpace(0, 255, shape=(240, 320, 3), dtype=np.uint8)])
    state_space = spaces.BoxSpace(np.deg2rad([-30., 45.]), np.deg2rad([30., 75.]))
    sensor_names = ['image']
    pr2_env = Pr2Env(action_space, observation_space, state_space, sensor_names)

    import IPython as ipy; ipy.embed()


if __name__ == "__main__":
    main()
