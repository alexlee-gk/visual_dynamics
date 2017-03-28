import argparse

import numpy as np
import yaml

from visual_dynamics.utils.config import Python2to3Loader
from visual_dynamics.utils.config import from_config, from_yaml
from visual_dynamics.utils.container import ImageDataContainer
from visual_dynamics.utils.rl_util import do_rollouts, discount_returns, FeaturePredictorServoingImageVisualizer
from visual_dynamics.utils.transformer import extract_image_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm_fname', type=str, nargs='*')
    parser.add_argument('--start_traj_iter', '-s', type=int, default=0)
    parser.add_argument('--num_trajs', '-n', type=int, default=100, metavar='N', help='number of trajectories')
    parser.add_argument('--num_steps', '-t', type=int, default=100, metavar='T', help='maximum number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--record_file', '-r', type=str)
    parser.add_argument('--reset_states_fname', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_fname', '-o', type=str)
    parser.add_argument('--observations_dir', '-d', type=str)
    parser.add_argument('--use_last', action='store_true')
    args = parser.parse_args()

    algorithm_configs = []
    for algorithm_fname in args.algorithm_fname:
        with open(algorithm_fname) as algorithm_file:
            algorithm_config = yaml.load(algorithm_file, Loader=Python2to3Loader)
        algorithm_configs.append(algorithm_config)

    best_return, algorithm_config = max(zip([max(algorithm_config['mean_returns']) for algorithm_config in algorithm_configs], algorithm_configs))

    if args.reset_states_fname is None:
        reset_states = [None] * args.num_trajs
    else:
        with open(args.reset_states_fname, 'r') as reset_state_file:
            reset_state_config = from_yaml(reset_state_file)
        algorithm_config['env']['env']['car_model_names'] = reset_state_config['environment_config']['car_model_names']
        reset_states = reset_state_config['reset_states']
        args.num_trajs = min(args.num_trajs, len(reset_states))

    alg = from_config(algorithm_config)
    env = alg.env
    servoing_pol = alg.servoing_pol

    if args.use_last:
        print("using parameters of the last iteration")
        best_return, best_theta = list(zip(alg.mean_returns, alg.thetas))[-1]
    else:
        print("using parameters based on best returns")
        best_return, best_theta = max(zip(alg.mean_returns, alg.thetas))
    print(best_return)
    servoing_pol.theta = best_theta
    print(servoing_pol.theta)

    if args.output_fname:
        import csv
        with open(args.output_fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(best_return)])
            writer.writerow(map(str, servoing_pol.theta))

    image_transformer = extract_image_transformer(dict(servoing_pol.predictor.transformers))
    image_visualizer = FeaturePredictorServoingImageVisualizer(servoing_pol.predictor, visualize=args.visualize)
    if args.observations_dir:
        _, observations, _, rewards = do_rollouts(env, servoing_pol, args.num_trajs, args.num_steps,
                                                  image_visualizer=image_visualizer,
                                                  verbose=args.verbose, seeds=np.arange(args.num_trajs),
                                                  reset_states=reset_states, cv2_record_file=args.record_file,
                                                  image_transformer=image_transformer)
        container = None
        for traj_iter, observations_ in enumerate(observations):
            for step_iter, obs in enumerate(observations_):
                if container is None:
                    container = ImageDataContainer(args.observations_dir, 'x')
                    container.reserve(list(obs.keys()), (args.num_trajs, args.num_steps + 1))
                container.add_datum(traj_iter, step_iter, **obs)
        container.close()
    else:
        rewards = do_rollouts(env, servoing_pol, args.num_trajs, args.num_steps,
                              image_visualizer=image_visualizer,
                              verbose=args.verbose, seeds=np.arange(args.num_trajs),
                              reset_states=reset_states, cv2_record_file=args.record_file,
                              image_transformer=image_transformer, ret_rewards_only=True)

    if args.output_fname:
        import csv
        discounted_returns = discount_returns(rewards, alg.gamma)
        returns = discount_returns(rewards, 1.0)
        with open(args.output_fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(map(str, discounted_returns))
            writer.writerow([str(np.mean(discounted_returns)), str(np.std(discounted_returns) / np.sqrt(len(discounted_returns)))])
            writer.writerow(map(str, returns))
            writer.writerow([str(np.mean(returns)), str(np.std(returns) / np.sqrt(len(returns)))])


if __name__ == '__main__':
    main()
