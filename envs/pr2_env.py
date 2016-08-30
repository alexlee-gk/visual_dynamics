import numpy as np
import rospy
from envs import RosEnv
from pr2 import PR2, camera_sensor
import spaces


class Pr2Env(RosEnv):
    def __init__(self, action_space, observation_space, state_space, sensor_names):
        super(Pr2Env, self).__init__(action_space, observation_space, state_space, sensor_names)
        self.pr2 = PR2.PR2()
        self.pr2.larm.goto_posture('side')
        self.pr2.rarm.goto_posture('side')
        self.pr2.torso.go_down()
        self.pr2.head.set_pan_tilt(*((self.state_space.low + self.state_space.high) / 2.0))

        self.rgb_camera_sensor = camera_sensor.CameraSensor()

    def step(self, action):
        # update action to be within the action space
        if not self.action_space.contains(action):
            action = self.action_space.clip(action, out=action)

        pan_tilt_angles = self.pr2.head.get_joint_positions()

        action[:] = self.state_space.clip(pan_tilt_angles + action) - pan_tilt_angles

        self.pr2.head.command_pan_tilt_vel(*action)
        rospy.sleep(.1)

    def get_state(self):
        return self.pr2.head.get_joint_positions()

    def reset(self, state=None):
        if state is None:
            state = self.state_space.sample()
        self.pr2.head.goto_joint_positions(state)
        rospy.sleep(1.0)

    def observe(self):
        obs = []
        for sensor_name in self.sensor_names:
            if sensor_name == 'image':
                observation = self.rgb_camera_sensor.observe()
            else:
                raise ValueError('Unknown sensor name %s' % sensor_name)
            obs.append(observation.copy())
        return obs

    def render(self):
        pass

    def _get_config(self):
        config = super(Pr2Env, self)._get_config()
        return config


def main():
    rospy.init_node('camera_sensor', anonymous=True)

    action_space = spaces.BoxSpace(np.deg2rad([-5., -5.]), np.deg2rad([5., 5.]))
    observation_space = spaces.Tuple([spaces.BoxSpace(0, 255, shape=(240, 320, 3), dtype=np.uint8)])
    state_space = spaces.BoxSpace(np.deg2rad([-30., 45.]), np.deg2rad([30., 75.]))
    sensor_names = ['image']
    pr2_env = Pr2Env(action_space, observation_space, state_space, sensor_names)

    import IPython as ipy; ipy.embed()


if __name__ == "__main__":
    main()
