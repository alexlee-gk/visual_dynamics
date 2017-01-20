import numpy as np
try:
    import rospy
    import sensor_msgs.msg as sensor_msgs
    import geometry_msgs.msg as geometry_msgs
    import cv_bridge
    import tf
except ImportError:
    pass
from envs import RosEnv
from spaces import TranslationAxisAngleSpace
from policy import Policy, PositionBasedServoingPolicy
from policy.quad_target_policy import xyz_to_hra, hra_to_xyz
from utils import transformations


class QuadRosEnv(RosEnv):
    def __init__(self, dt=None):
        self.dt = dt or 0.1
        self.sensor_names = ['quad_to_obj_pos', 'quad_to_obj_rot', 'image']

        # self.quad_to_obj_pos = np.zeros(3)
        # self.quad_to_obj_rot = np.array([0, 0, 0, 1])
        # self.image = np.zeros((32, 32), dtype=np.uint8)
        # self.quad_pos = np.zeros(3)
        # self.quad_rot = np.array([0, 0, 0, 1])
        self.quad_to_obj_pos = None
        self.quad_to_obj_rot = None
        self.image = None
        self.quad_pos = None
        self.quad_rot = None

        self.camera_control_pub = rospy.Publisher("/bebop/camera_control", geometry_msgs.Twist, queue_size=1, latch=True)
        self.cmd_vel_pub = rospy.Publisher("/vservo/cmd_vel", geometry_msgs.Twist, queue_size=1)

        # set camera's tilt angle to -35 degrees
        twist_msg = geometry_msgs.Twist(
            linear=geometry_msgs.Point(0., 0., 0.),
            angular=geometry_msgs.Point(0., -35., 0.)
        )
        self.camera_control_pub.publish(twist_msg)

        self.listener = tf.TransformListener()
        self.image_sub = rospy.Subscriber("/bebop/image_raw", sensor_msgs.Image, callback=self._image_callback)
        self.cv_bridge = cv_bridge.CvBridge()

        self._action_space = TranslationAxisAngleSpace(-np.ones(4), np.ones(4), axis=np.array([0, 0, 1]))

        self.rate = rospy.Rate(1.0 / self.dt)

        rospy.sleep(1)

    def _image_callback(self, image_msg):
        try:
            self.quad_to_obj_pos, self.quad_to_obj_rot = self.listener.lookupTransform('bebop', 'car', image_msg.header.stamp)
            self.quad_pos, self.quad_rot = self.listener.lookupTransform('world', 'bebop', image_msg.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
        self.image = self.cv_bridge.imgmsg_to_cv2(image_msg)

    def step(self, action):
        twist_msg = geometry_msgs.Twist(
            linear=geometry_msgs.Point(action[1], -action[0], action[2]),  # forward, left, ascend
            angular=geometry_msgs.Point(0., 0., action[3])
        )
        self.cmd_vel_pub.publish(twist_msg)
        self.rate.sleep()
        while self.quad_to_obj_pos is None or \
                self.quad_to_obj_rot is None or \
                self.image is None:
            rospy.sleep(.1)
        obs = np.array(self.quad_to_obj_pos), np.array(self.quad_to_obj_rot), self.image
        return obs, None, None, None

    def reset(self, state=None):
        while self.quad_to_obj_pos is None or \
                self.quad_to_obj_rot is None or \
                self.image is None:
            rospy.sleep(.1)
        obs = np.array(self.quad_to_obj_pos), np.array(self.quad_to_obj_rot), self.image
        return obs

    def get_state(self):
        while self.quad_pos is None or self.quad_rot is None:
            rospy.sleep(.1)
        quad_pos, quad_rot = self.quad_pos, self.quad_rot
        quad_aa = transformations.axis_angle_from_quaternion(np.r_[quad_rot[3], quad_rot[:3]])
        up = np.array([0, 0, 1])
        return np.r_[quad_pos, quad_aa.dot(up)]

    @property
    def action_space(self):
        return self._action_space

    def get_relative_target_position(self):
        while self.quad_to_obj_pos is None:
            rospy.sleep(.1)
        return self.quad_to_obj_pos

    def is_in_view(self):
        # TODO
        return True

    def _get_config(self):
        config = super(QuadRosEnv, self)._get_config()
        config.update({'dt': self.dt})
        return config


class QuadTargetPolicy(Policy):
    def __init__(self, lambda_, height_limits, angle_limits, radius_limits=None, straight_trajectory=True, tightness=0.1):
        self.height_limits = list(height_limits)
        self.radius_limits = list(radius_limits) if radius_limits is not None else radius_limits
        self.angle_limits = list(angle_limits)
        self.tightness = tightness
        self.target_hra = None
        self.pbvs_pol = PositionBasedServoingPolicy(lambda_, None, straight_trajectory=straight_trajectory)

    def act(self, obs):
        quad_to_obj_pos, quad_to_obj_rot = obs[:2]
        quad_to_obj_T = transformations.quaternion_matrix(np.r_[quad_to_obj_rot[3], quad_to_obj_rot[:3]])
        quad_to_obj_T[:3, 3] = quad_to_obj_pos
        obj_to_quad_T = transformations.inverse_matrix(quad_to_obj_T)

        if self.tightness == 1.0:
            des_offset_hra = self.target_hra
        else:
            offset = obj_to_quad_T[:3, 3]
            offset_hra = xyz_to_hra(offset)
            target_hra = self.target_hra.copy()
            offset_hra[-1], target_hra[-1] = np.unwrap([offset_hra[-1], target_hra[-1]])
            des_offset_hra = (1 - self.tightness) * offset_hra + self.tightness * target_hra
        des_offset = hra_to_xyz(des_offset_hra)
        des_obj_to_quad_T = transformations.rotation_matrix(des_offset_hra[2], np.array([0, 0, 1]))
        des_obj_to_quad_T[:3, 3] = des_offset
        self.pbvs_pol.target_to_obj_T = transformations.inverse_matrix(des_obj_to_quad_T)
        return self.pbvs_pol.act(obs)

    def reset(self):
        self.target_hra = self.sample_hra()
        return None

    def sample_hra(self):
        height = np.random.uniform(*self.height_limits)
        radius = np.random.uniform(*self.radius_limits) if self.radius_limits is not None else height / np.tan(np.deg2rad(35))
        angle = np.random.uniform(*self.angle_limits)
        return np.array([height, radius, angle])


def main():
    from policy import PositionBasedServoingPolicy

    rospy.init_node("quad_ros_env", anonymous=True)

    env = QuadRosEnv()

    target_to_obj_T = np.eye(4)
    target_to_obj_T[:3, 3] = np.array([0., 1.0 / np.tan(np.deg2rad(35)), -1.0])
    pol = PositionBasedServoingPolicy(1.0, target_to_obj_T)
    # pol = QuadTargetPolicy(1.0, (1.0, 1.0), (np.pi / 2, np.pi / 2), tightness=0.4)
    # pol = QuadTargetPolicy(1.0, (1.0, 1.0), (0, 0), tightness=0.2)
    # pol = QuadTargetPolicy(1.0, (1.0, 1.2), (-np.pi / 2, np.pi / 2), tightness=0.5)

    obs = env.reset()
    while True:
        try:
            action = pol.act(obs)
            print(action)
            obs, _, _, _ = env.step(action)
            if rospy.is_shutdown():
                break
        except KeyboardInterrupt:
            break
    return


    num_trajs = 10
    num_steps = 100
    done = False
    for traj_iter in range(num_trajs):
        try:
            state = pol.reset()
            obs = env.reset(state)
            for step_iter in range(num_steps):
                try:
                    action = pol.act(obs)
                    print(action)
                    obs, _, _, _ = env.step(action)
                except KeyboardInterrupt:
                    done = True
                    break
                if done or rospy.is_shutdown():
                    break
            if done or rospy.is_shutdown():
                break
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
