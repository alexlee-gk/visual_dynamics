from __future__ import division, print_function

import argparse
import yaml
from citysim3d.envs import ServoingEnv

from visual_dynamics import policies
from visual_dynamics import utils
from visual_dynamics import envs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    parser.add_argument('--target_distance', '-d', type=int, default=0)
    parser.add_argument('--feature_inds', '-i', type=str, help='inds of subset of features to use')
    parser.add_argument('--w_init', type=float, default=10.0)
    parser.add_argument('--lambda_init', type=float, default=1.0)
    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)

    if issubclass(predictor_config['environment_config']['class'], envs.Panda3dEnv):
        utils.transfer_image_transformer(predictor_config)

    predictor = utils.from_config(predictor_config)
    if args.feature_inds:
        args.feature_inds = [int(ind) for ind in args.feature_inds]
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]

    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("visual_servoing")
    env = utils.from_config(predictor.environment_config)
    if not isinstance(env, ServoingEnv):
        env = ServoingEnv(env, max_time_steps=args.num_steps)

    pol = policies.ServoingPolicy(predictor, alpha=1.0, lambda_=args.lambda_init, w=args.w_init)

    if args.record_file and not args.visualize:
        args.visualize = 1
    if args.visualize:
        image_visualizer = utils.FeaturePredictorServoingImageVisualizer(predictor, visualize=args.visualize)
        utils.do_rollouts(env, pol, args.num_trajs, args.num_steps,
                          target_distance=args.target_distance,
                          output_dir=args.output_dir,
                          image_visualizer=image_visualizer,
                          record_file=args.record_file,
                          gamma=args.gamma,
                          verbose=True)


if __name__ == "__main__":
    main()
