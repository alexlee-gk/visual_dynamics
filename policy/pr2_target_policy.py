import numpy as np
try:
    import rospy
except:
    pass
from envs import Pr2Env
from policy import TargetPolicy
import utils.transformations as tf
from pr2 import berkeley_pr2


class Pr2TargetPolicy(TargetPolicy):
    def __init__(self, env, frame_id, offset):
        """
        This policy points the camera to the offset in the target frame
        Args:
            env: Pr2Env
            frame_id: frame id of the target
            offset: offset relative to the target frame
        """
        self.env = env
        if not isinstance(self.env, Pr2Env):
            raise ValueError("env should be of type Pr2Env but instead it is of type %s" % type(self.env))
        self.frame_id = frame_id
        self.offset = offset

    def act(self, obs):
        return (self.get_target_state() - self.env.get_state()) / self.env.dt

    def reset(self):
        return self.get_target_state()

    def get_target_state(self):
        self.env.pr2.update_rave()
        # target transform in world coordinates
        target_T = self.env.pr2.robot.GetLink(self.frame_id).GetTransform()
        # target offset relative to target
        target_to_offset_T = tf.translation_matrix(self.offset)
        # target offset in world coordinates
        offset_T = target_T.dot(target_to_offset_T)
        # camera transform in world coordinates
        camera_T = berkeley_pr2.get_kinect_transform(self.env.pr2.robot)
        # pointing axis
        ax = offset_T[:3, 3] - camera_T[:3, 3]
        pan = np.arctan(ax[1] / ax[0])
        tilt = np.arcsin(-ax[2] / np.linalg.norm(ax))
        return pan, tilt

    def _get_config(self):
        config = super(Pr2TargetPolicy, self)._get_config()
        config.update({'env': self.env,
                       'frame_id': self.frame_id,
                       'offset': self.offset})
        return config
