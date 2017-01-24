import argparse
import numpy as np
import rospy

from visual_dynamics import utils, policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default="r")
    args = parser.parse_args()

    rospy.init_node("pr2_target_policy")
    with open('config/environment/pr2.yaml') as yaml_string:
        pr2_env = utils.from_yaml(yaml_string)

    tool_link_name = "%s_gripper_tool_frame" % args.lr
    pol = policies.Pr2TargetPolicy(pr2_env, tool_link_name, np.zeros(3))


    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        state = pr2_env.get_state()
        target_state = pol.get_target_state()
        action = (target_state - state) / pr2_env.dt
        pr2_env.step(action)
        # pr2_env.reset(pol.get_target_state())
        rate.sleep()


if __name__ == "__main__":
    main()
