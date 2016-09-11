import numpy as np
try:
    import rospy
except:
    pass
import spaces
from policy import Pr2TargetPolicy
from pr2 import planning, pr2_trajectories


class Pr2MovingArmTargetPolicy(Pr2TargetPolicy):
    def __init__(self, env, frame_id, offset, lr='r', gripper_state_space=None):
        """
        This policy points the camera to the offset in the target frame
        Args:
            env: Pr2Env
            frame_id: frame id of the target
            offset: offset relative to the target frame
            lr: 'l' for left arm and 'r' for right arm
            gripper_state_space: space of the target position for the gripper tool frame
        """
        super(Pr2MovingArmTargetPolicy, self).__init__(env, frame_id, offset)
        self.pr2 = self.env.pr2
        self.lr = lr
        default_gripper_state_spaces = dict(r=spaces.BoxSpace(np.array([.5, -.3, .8]),
                                                              np.array([.7, .3, 1])),
                                            l=spaces.BoxSpace(np.array([.5, -.3, .8]),
                                                              np.array([.7, .3, 1])))
        self.gripper_state_space = gripper_state_space or default_gripper_state_spaces[self.lr]
        target_pos = (self.gripper_state_space.low + self.gripper_state_space.high) / 2.0
        self.start_arm_trajectory(target_pos, wait=True, speed_factor=1)

    def act(self, obs):
        if not self.env.pr2.is_moving():
            self.start_arm_trajectory()
        return super(Pr2MovingArmTargetPolicy, self).act(obs)

    def reset(self):
        if not self.env.pr2.is_moving():
            self.start_arm_trajectory()
        return super(Pr2MovingArmTargetPolicy, self).reset()

    def start_arm_trajectory(self, target_pos=None, wait=False, speed_factor=.1):
        if target_pos is None:
            target_pos = self.gripper_state_space.sample()
        if isinstance(target_pos, np.ndarray):
            target_pos = target_pos.tolist()
        self.pr2.update_rave()
        traj = planning.plan_up_trajectory(self.pr2.robot, self.lr, target_pos)
        bodypart2traj = {"%sarm" % self.lr: traj}
        pr2_trajectories.follow_body_traj(self.pr2, bodypart2traj, wait=wait, speed_factor=speed_factor)

    def _get_config(self):
        config = super(Pr2MovingArmTargetPolicy, self)._get_config()
        config.update({'lr': self.lr})
        return config
