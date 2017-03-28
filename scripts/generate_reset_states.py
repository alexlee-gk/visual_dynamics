import argparse
import yaml

from visual_dynamics import envs
from visual_dynamics.utils.config import from_config, to_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_fname', type=str, help='config file with environment arguments')
    parser.add_argument('--reset_states_fname', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    args = parser.parse_args()

    with open(args.env_fname) as env_file:
        env_config = yaml.load(env_file)
        if issubclass(env_config['class'], envs.RosEnv):
            import rospy
            rospy.init_node("generate_reset_states")
        env = from_config(env_config)

    reset_states = []
    for traj_iter in range(args.num_trajs):
        env.reset()
        if args.visualize:
            env.render()
        reset_states.append(env.get_state())

    if args.reset_states_fname:
        with open(args.reset_states_fname, 'w') as reset_state_file:
            to_yaml(dict(environment_config=env.get_config(),
                         reset_states=[reset_state.tolist() for reset_state in reset_states]),
                    reset_state_file)


if __name__ == '__main__':
    main()
