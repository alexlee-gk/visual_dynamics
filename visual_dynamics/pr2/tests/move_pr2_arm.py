import argparse
import numpy as np
import rospy
import trajoptpy
from pr2 import pr2_trajectories, yes_or_no

from visual_dynamics import utils
from visual_dynamics.pr2 import planning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default="r")
    parser.add_argument("--use_cost", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no_prompt", action="store_true")
    args = parser.parse_args()

    trajoptpy.SetInteractive(args.interactive)  # pause every iteration, until you press 'p'. Press escape to disable further plotting

    rospy.init_node("pr2_arm")
    with open('config/environment/pr2.yaml') as yaml_string:
        pr2_env = utils.from_yaml(yaml_string)

    if args.lr == 'r':
        min_target_pos = np.array([.5, -.5, .8])
        max_target_pos = np.array([.7, .2, 1])
    else:
        min_target_pos = np.array([.5, -.2, .8])
        max_target_pos = np.array([.7, .5, 1])

    while not rospy.is_shutdown():
        if args.no_prompt or yes_or_no.yes_or_no("execute?"):
            target_pos = utils.sample_interval(min_target_pos, max_target_pos).tolist()
            pr2_env.pr2.update_rave()
            traj = planning.plan_up_trajectory(pr2_env.pr2.robot, args.lr, target_pos, use_cost=args.use_cost)
            bodypart2traj = {"%sarm" % args.lr: traj}
            pr2_trajectories.follow_body_traj(pr2_env.pr2, bodypart2traj, speed_factor=0.1)


if __name__ == "__main__":
    main()
