import numpy as np
import utils.transformations as tf
from policy import Policy


def hra_to_xyz(hra):
    height, radius, angle = hra
    return np.array([radius * np.sin(angle), -radius * np.cos(angle), height])


def xyz_to_hra(xyz):
    return np.array([xyz[2], np.linalg.norm(xyz[:2]), np.arctan2(xyz[0], -xyz[1])])


class QuadTargetPolicy(Policy):
    def __init__(self, env, height_limits, angle_limits, radius_limits=None, tightness=0.1):
        self.env = env
        self.height_limits = list(height_limits)
        self.radius_limits = list(radius_limits) if radius_limits is not None else radius_limits
        self.angle_limits = list(angle_limits)
        self.tightness = tightness
        self.target_hra = None

    def act(self, obs):
        quad_pos = np.array(self.env.quad_node.getPos())
        if self.tightness == 1.0:
            des_offset_hra = self.target_hra
        else:
            hor_car_T = self.env.hor_car_T
            hor_car_inv_T = tf.inverse_matrix(hor_car_T)
            offset = hor_car_inv_T[:3, :3].dot(quad_pos) + hor_car_inv_T[:3, 3]
            offset_hra = xyz_to_hra(offset)
            target_hra = self.target_hra.copy()
            offset_hra[-1], target_hra[-1] = np.unwrap([offset_hra[-1], target_hra[-1]])
            des_offset_hra = (1 - self.tightness) * offset_hra + self.tightness * target_hra
        des_offset = hra_to_xyz(des_offset_hra)

        # desired quad transform in world coordinates
        des_quad_T = tf.pose_matrix(*self.env.compute_desired_quad_pos_quat(offset=des_offset)[::-1])
        # quad transform in world coordinates
        quad_T = tf.pose_matrix(self.env.quad_node.getQuat(), self.env.quad_node.getPos())
        # desired quad transform relative to the quad
        quad_to_des_quad_T = tf.inverse_matrix(quad_T).dot(des_quad_T)

        linear_vel, angular_vel = np.split(tf.position_axis_angle_from_matrix(quad_to_des_quad_T) / self.env.dt, [3])
        if self.env.action_space.axis is not None:
            angular_vel = angular_vel.dot(self.env.action_space.axis)
        action = np.append(linear_vel, angular_vel)
        return action

    def reset(self):
        # sample reset and target offsets
        reset_hra = self.sample_hra()
        self.target_hra = self.sample_hra()
        # save original state of the car environment
        orig_car_state = self.env.car_env.get_state()
        # reset the car environment to a random state
        self.env.car_env.reset()
        car_state = self.env.car_env.get_state()
        # compute the quad transform for the new state of the car environment
        des_quad_T = tf.pose_matrix(*self.env.compute_desired_quad_pos_quat(offset=hra_to_xyz(reset_hra))[::-1])
        # restore original state of the car environment
        self.env.car_env.reset(orig_car_state)
        return np.concatenate([tf.position_axis_angle_from_matrix(des_quad_T), car_state])

    def sample_hra(self):
        height = np.random.uniform(*self.height_limits)
        radius = np.random.uniform(*self.radius_limits) if self.radius_limits is not None else np.sqrt(3) * height
        angle = np.random.uniform(*self.angle_limits)
        return np.array([height, radius, angle])

    def _get_config(self):
        config = super(QuadTargetPolicy, self)._get_config()
        config.update({'env': self.env,
                       'height_limits': self.height_limits,
                       'angle_limits': self.angle_limits,
                       'radius_limits': self.radius_limits,
                       'tightness': self.tightness})
        return config
