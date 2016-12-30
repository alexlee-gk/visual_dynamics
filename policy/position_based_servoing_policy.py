import numpy as np
from policy import Policy
import utils.transformations as tf


class PositionBasedServoingPolicy(Policy):
    def __init__(self, lambda_, target_to_obj_T, straight_trajectory=True):
        """
        target_to_obj_T is in inertial reference frame
        """
        self.lambda_ = lambda_
        self.target_to_obj_T = np.asarray(target_to_obj_T)
        self.straight_trajectory = straight_trajectory

    def act(self, obs):
        curr_to_obj_pos, curr_to_obj_rot = obs[:2]
        curr_to_obj_T = tf.quaternion_matrix(np.r_[curr_to_obj_rot[3], curr_to_obj_rot[:3]])
        curr_to_obj_T[:3, 3] = curr_to_obj_pos
        target_to_curr_T = self.target_to_obj_T.dot(tf.inverse_matrix(curr_to_obj_T))
        target_to_curr_aa = tf.axis_angle_from_matrix(target_to_curr_T)
        # project rotation so that it rotates around the up vector
        up = np.array([0, 0, 1])
        target_to_curr_aa = target_to_curr_aa.dot(up) * up
        target_to_curr_T[:3, :3] = tf.matrix_from_axis_angle(target_to_curr_aa)[:3, :3]
        if self.straight_trajectory:
            linear_vel = -self.lambda_ * target_to_curr_T[:3, :3].T.dot(target_to_curr_T[:3, 3])
            angular_vel = -self.lambda_ * target_to_curr_aa.dot(up)
        else:
            linear_vel = -self.lambda_ * ((self.target_to_obj_T[:3, 3] - curr_to_obj_pos) +  np.cross(curr_to_obj_pos, target_to_curr_aa))
            angular_vel = -self.lambda_ * target_to_curr_aa.dot(up)
        action = np.r_[linear_vel, angular_vel]
        return action

    def reset(self):
        return None

    def _get_config(self):
        config = super(PositionBasedServoingPolicy, self)._get_config()
        config.update({'lambda_': self.lambda_,
                       'target_to_obj_T': self.target_to_obj_T.tolist(),
                       'straight_trajectory': self.straight_trajectory})
        return config
