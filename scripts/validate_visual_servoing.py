from __future__ import division, print_function

import csv

import argparse
import numpy as np
import yaml
from citysim3d.envs import ServoingEnv

from visual_dynamics import envs
from visual_dynamics import policies
from visual_dynamics.utils.config import from_config
from visual_dynamics.utils.rl_util import do_rollouts, discount_returns, FeaturePredictorServoingImageVisualizer
from visual_dynamics.utils.transformer import transfer_image_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--target_distance', '-d', type=int, default=0)
    parser.add_argument('--feature_inds', '-i', type=str, help='inds of subset of features to use')
    parser.add_argument('--w_inits', nargs='+', type=float, default=list(range(1, 101)))
    parser.add_argument('--lambda_inits', nargs='+', type=float, default=[1.0])
    parser.add_argument('--output_fname', '-o', type=str)
    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)

    if issubclass(predictor_config['environment_config']['class'], envs.Panda3dEnv):
        transfer_image_transformer(predictor_config)

    predictor = from_config(predictor_config)
    if args.feature_inds:
        args.feature_inds = [int(ind) for ind in args.feature_inds]
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]

    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("validate_visual_servoing")
    env = from_config(predictor.environment_config)
    if not isinstance(env, ServoingEnv):
        env = ServoingEnv(env, max_time_steps=args.num_steps)

    if args.visualize:
        image_visualizer = FeaturePredictorServoingImageVisualizer(predictor, visualize=args.visualize)
    else:
        image_visualizer = None

    header_format = '{:>15}\t{:>15}\t{:>15}\t{:>15}'
    row_format = '{:>15.2f}\t{:>15.2f}\t{:>15.4f}\t{:>15.4f}'
    print(header_format.format('w_init', 'lambda_init', 'mean', 'standard error'))
    print('=' * 60)

    if args.output_fname:
        with open(args.output_fname, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['w_init', 'lambda_init', 'mean', 'standard error'])

    for w_init in args.w_inits:
        for lambda_init in args.lambda_inits:
            pol = policies.ServoingPolicy(predictor, alpha=1.0, lambda_=lambda_init, w=w_init)
            _, _, _, rewards = do_rollouts(env, pol, args.num_trajs, args.num_steps,
                                           target_distance=args.target_distance,
                                           image_visualizer=image_visualizer,
                                           gamma=args.gamma,
                                           seeds=np.arange(args.num_trajs))
            discounted_returns = discount_returns(rewards, args.gamma)
            row_values = [w_init, lambda_init, np.mean(discounted_returns), np.std(discounted_returns) / np.sqrt(len(discounted_returns))]
            print(row_format.format(*row_values))

            if args.output_fname:
                with open(args.output_fname, 'ab') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([str(value) for value in (row_values + list(discounted_returns))])


if __name__ == "__main__":
    main()
