from __future__ import division, print_function

import argparse
import colorsys
import os

import yaml

from visual_dynamics import envs
from visual_dynamics import policies
from visual_dynamics.envs import ServoingEnv
from visual_dynamics.utils.config import from_config, from_yaml
from visual_dynamics.utils.rl_util import do_rollouts, FeaturePredictorServoingImageVisualizer
from visual_dynamics.utils.transformer import transfer_image_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('algorithm_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    parser.add_argument('--feature_inds', '-i', type=int, nargs='+', help='inds of subset of features to use')
    parser.add_argument('--w_init', type=float, default=10.0)
    parser.add_argument('--lambda_init', type=float, default=1.0)
    parser.add_argument('--image_transformer_fname', type=str)
    parser.add_argument('--car_model_names', type=str, nargs='+')
    parser.add_argument('--car_ambient_light_color', '--car_color', type=float, nargs='+', help='H[S[V]]')

    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)

    if args.image_transformer_fname:
        with open(args.image_transformer_fname) as image_transformer_file:
            image_transformer = from_yaml(image_transformer_file)
    else:
        image_transformer = None
    if issubclass(predictor_config['environment_config']['class'], envs.Panda3dEnv):
        transfer_image_transformer(predictor_config, image_transformer)

    predictor = from_config(predictor_config)
    if args.feature_inds:
        args.feature_inds = [int(ind) for ind in args.feature_inds]
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]

    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("learn_visual_servoing")
    if args.car_model_names:
        predictor.environment_config['car_model_names'] = args.car_model_names

    env = from_config(predictor.environment_config)
    if not isinstance(env, ServoingEnv):
        env = ServoingEnv(env)
    if args.car_ambient_light_color:
        args.car_ambient_light_color.extend([1.0] * (3 - len(args.car_ambient_light_color)))
        assert len(args.car_ambient_light_color) == 3
        color = colorsys.hsv_to_rgb(*args.car_ambient_light_color) + (1,)
        color_id = '_'.join([str(c) for c in color[:3]])
        car_light_node = None
        for child in env.app.render.getChildren():
            if child.getName() == 'car_ambient_light':
                car_light_node = child
        car_light_node.node().setColor(tuple(color))

    servoing_pol = policies.TheanoServoingPolicy(predictor, alpha=1.0, lambda_=args.lambda_init, w=args.w_init)

    with open(args.algorithm_fname) as algorithm_file:
        algorithm_config = yaml.load(algorithm_file)
    algorithm_config['env'] = env
    algorithm_config['servoing_pol'] = servoing_pol

    if 'snapshot_prefix' not in algorithm_config:
        snapshot_prefix_paths = [os.path.split(args.predictor_fname)[0],
                                 os.path.splitext(os.path.split(args.algorithm_fname)[1])[0]]
        if args.car_model_names:
            snapshot_prefix_paths.append(''.join(args.car_model_names))
        if args.car_ambient_light_color:
            snapshot_prefix_paths.append(color_id)
        snapshot_prefix_paths.append('')
        snapshot_prefix = os.path.join(*snapshot_prefix_paths)
        algorithm_config['snapshot_prefix'] = snapshot_prefix

    alg = from_config(algorithm_config)
    alg.run()

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
