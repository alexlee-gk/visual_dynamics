from __future__ import division, print_function
import argparse
import yaml
import numpy as np
import scipy.stats
import lasagne
import theano
import theano.tensor as T
import time
import cv2
import envs
import policy
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer
from gui.loss_plotter import LossPlotter
import _tkinter
import pickle
from visual_servoing_weighted import FittedQIterationAlgorithm, do_rollout
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('reset_states_fname', type=str)
    parser.add_argument('--pol_fname', '--pol', type=str, help='config file with policy arguments for overriding')
    parser.add_argument('--car_model_name', type=str, nargs='+', help='car model name(s) for overriding')
    parser.add_argument('--num_steps', '-t', type=int, default=100, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--w_init', type=float, default=None)
    parser.add_argument('--lambda_init', type=float, default=None)
    parser.add_argument('--theta_init', type=float, nargs='+', default=None)
    parser.add_argument('--feature_inds', '-i', type=int, nargs='+', help='inds of subset of features to use')
    parser.add_argument('--target_distance', '-d', type=float, default=0.0)

    args = parser.parse_args()

    with open(args.reset_states_fname, 'rb') as states_file:
        reset_states = pickle.load(states_file)
        if isinstance(reset_states, dict):
            car_model_name = reset_states['car_model_name']
            reset_states = reset_states['reset_states']
        else:
            car_model_name = None
    gamma = 0.9

    predictor = utils.from_yaml(open(args.predictor_fname))
    if args.feature_inds:
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]
    environment_config = predictor.environment_config
    if args.car_model_name is not None:
        environment_config['car_model_name'] = args.car_model_name
    elif car_model_name is not None:
        environment_config['car_model_name'] = car_model_name

    env = utils.from_config(environment_config)

    if args.pol_fname:
        with open(args.pol_fname) as yaml_string:
            policy_config = yaml.load(yaml_string)
    else:
        policy_config = predictor.policy_config
    replace_config = {'env': env,
                      'action_space': env.action_space,
                      'state_space': env.state_space}
    try:
        replace_config['target_env'] = env.car_env
    except AttributeError:
        pass
    pol = utils.from_config(policy_config, replace_config=replace_config)
    if isinstance(pol, policy.RandomPolicy):
        target_pol = pol
        random_pol = pol
    else:
        assert len(pol.policies) == 2
        target_pol, random_pol = pol.policies
        assert pol.reset_probs == [1, 0]
    del pol
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(random_pol, policy.RandomPolicy)
    servoing_pol = policy.TheanoServoingPolicy(predictor, alpha=1.0, lambda_=1.0)
    servoing_pol.w = 10.0 * np.ones_like(servoing_pol.w)
    if args.w_init is not None:
        servoing_pol.w = args.w_init * np.ones_like(servoing_pol.w)
    if args.lambda_init is not None:
        servoing_pol.lambda_ = args.lambda_init * np.ones_like(servoing_pol.lambda_)
    if args.theta_init is not None:
        servoing_pol.w, servoing_pol.lambda_, _ = np.split(args.theta_init, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))

    np.random.seed(7)
    print(servoing_pol.w, servoing_pol.lambda_)
    test_costs = []
    for traj_iter, reset_state in enumerate(reset_states):
        print('test_rollouts. traj_iter', traj_iter)
        states, actions, rewards, next_states, target_actions = do_rollout(predictor, env, reset_state, servoing_pol,
                                                                           target_pol,
                                                                           args.num_steps,
                                                                           visualize=0,
                                                                           image_visualizer=None,
                                                                           target_distance=args.target_distance,
                                                                           quiet=True)
        test_cost = np.array(rewards).dot(gamma ** np.arange(len(rewards)))
        print(test_cost)
        test_costs.append(test_cost)

    cost_str_output = '\n'.join([str(test_cost) for test_cost in test_costs])
    print(cost_str_output)


if __name__ == '__main__':
    main()
