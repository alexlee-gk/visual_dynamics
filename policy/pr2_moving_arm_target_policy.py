import numpy as np
try:
    import rospy
except:
    pass
import spaces
from policy import Pr2TargetPolicy
from pr2 import planning, pr2_trajectories, berkeley_pr2


class Pr2MovingArmTargetPolicy(Pr2TargetPolicy):
    def __init__(self, env, frame_id, offset, lr='r', min_gripper_displacement=0.05, gripper_state_space=None):
        """
        This policy points the camera to the offset in the target frame
        Args:
            env: Pr2Env
            frame_id: frame id of the target
            offset: offset relative to the target frame
            lr: 'l' for left arm and 'r' for right arm
            min_gripper_displacement: minimum distance to move when sampling for a new gripper position 
            gripper_state_space: pan, tilt and distance of the target location for the gripper tool frame
        """
        super(Pr2MovingArmTargetPolicy, self).__init__(env, frame_id, offset)
        self.pr2 = self.env.pr2
        self.lr = lr
        self.min_gripper_displacement = min_gripper_displacement
        self.gripper_state_space = gripper_state_space or \
            spaces.BoxSpace(np.r_[self.env.state_space.low, .6],
                            np.r_[self.env.state_space.high, .8])
        target_gripper_state = (self.gripper_state_space.low + self.gripper_state_space.high) / 2.0
        self.start_arm_trajectory(self.target_pos_from_gripper_state(target_gripper_state),
                                  wait=True, speed_factor=1)

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
            while target_pos is None or \
                    np.linalg.norm(target_pos - curr_pos) < self.min_gripper_displacement:
                self.env.pr2.update_rave()
                curr_pos = self.env.pr2.robot.GetLink(self.frame_id).GetTransform()[:3, 3]
                gripper_state = self.gripper_state_space.sample()
                target_pos = self.target_pos_from_gripper_state(gripper_state)
        if isinstance(target_pos, np.ndarray):
            target_pos = target_pos.tolist()
        self.pr2.update_rave()
        traj = planning.plan_up_trajectory(self.pr2.robot, self.lr, target_pos)
        bodypart2traj = {"%sarm" % self.lr: traj}
        pr2_trajectories.follow_body_traj(self.pr2, bodypart2traj, wait=wait, speed_factor=speed_factor)

    def target_pos_from_gripper_state(self, gripper_state):
        pan, tilt, distance = gripper_state
        camera_T = berkeley_pr2.get_kinect_transform(self.env.pr2.robot)
        ax2 = -np.sin(tilt) * distance
        ax0 = -np.cos(pan) * ax2 / np.tan(tilt)
        ax1 = -np.sin(pan) * ax2 / np.tan(tilt)
        ax = np.array([ax0, ax1, ax2])
        target_pos = ax + camera_T[:3, 3]
        return target_pos

    def _get_config(self):
        config = super(Pr2MovingArmTargetPolicy, self)._get_config()
        config.update({'lr': self.lr})
        return config
