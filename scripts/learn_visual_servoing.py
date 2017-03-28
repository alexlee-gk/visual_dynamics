from __future__ import division, print_function

import argparse
import datetime
import os
import uuid

import dateutil.tz
import yaml

from visual_dynamics import envs
from visual_dynamics import policies
from visual_dynamics.envs import ServoingEnv
from visual_dynamics.utils.config import Python2to3Loader
from visual_dynamics.utils.config import from_config
from visual_dynamics.utils.rl_util import do_rollouts, FeaturePredictorServoingImageVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('algorithm_fname', type=str)
    parser.add_argument('--algorithm_init_fname', type=str, default=None)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    parser.add_argument('--cv2_record_file', type=str, default=None)
    parser.add_argument('--w_init', type=float, nargs='+', default=1.0)
    parser.add_argument('--lambda_init', type=float, nargs='+', default=1.0)

    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file, Loader=Python2to3Loader)

    predictor = from_config(predictor_config)

    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("learn_visual_servoing")

    env = from_config(predictor.environment_config)
    if not isinstance(env, ServoingEnv):
        env = ServoingEnv(env)

    servoing_pol = policies.TheanoServoingPolicy(predictor, alpha=1.0, lambda_=args.lambda_init, w=args.w_init)

    with open(args.algorithm_fname) as algorithm_file:
        algorithm_config = yaml.load(algorithm_file, Loader=Python2to3Loader)
    algorithm_config['env'] = env
    algorithm_config['servoing_pol'] = servoing_pol

    if 'snapshot_prefix' not in algorithm_config:
        snapshot_prefix_paths = [os.path.split(args.predictor_fname)[0],
                                 os.path.splitext(os.path.split(args.algorithm_fname)[1])[0]]
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        rand_id = str(uuid.uuid4())[:5]
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        snapshot_prefix_paths.append('%s_%s' % (timestamp, rand_id))
        snapshot_prefix_paths.append('')
        snapshot_prefix = os.path.join(*snapshot_prefix_paths)
        algorithm_config['snapshot_prefix'] = snapshot_prefix

    alg = from_config(algorithm_config)

    # TODO: concatenate an arbitrary number of algorithms
    if args.algorithm_init_fname:
        with open(args.algorithm_init_fname) as algorithm_init_file:
            algorithm_init_config = yaml.load(algorithm_init_file, Loader=Python2to3Loader)
        print("using parameters based on best returns")
        best_return, best_theta = max(zip(algorithm_init_config['mean_returns'], algorithm_init_config['thetas']))
        print(best_return)
        # servoing_pol.theta = best_theta
        servoing_pol.w = best_theta[:-len(servoing_pol.lambda_)]
        servoing_pol.lambda_ = best_theta[-len(servoing_pol.lambda_):]
        print(servoing_pol.theta)
    alg.run()
    print(servoing_pol.theta)

    if args.record_file and not args.visualize:
        args.visualize = 1
    if args.visualize:
        image_visualizer = FeaturePredictorServoingImageVisualizer(predictor, visualize=args.visualize)
        do_rollouts(env, servoing_pol, alg.num_trajs, alg.num_steps,
                    output_dir=args.output_dir,
                    image_visualizer=image_visualizer,
                    record_file=args.record_file,
                    verbose=True,
                    gamma=alg.gamma)


if __name__ == '__main__':
    main()
