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
    parser.add_argument('experiment_fname', type=str)
    parser.add_argument('reset_states_fname', type=str)
    parser.add_argument('--car_model_name', type=str, nargs='+', help='car model name(s) for overriding')
    parser.add_argument('--num_trajs_to_save', type=int)
    parser.add_argument('--num_steps', '-t', type=int, default=100, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--reset_after_target', type=int, default=0)
    parser.add_argument('--save_tpv', action='store_true', help='also save 3rd person view')
    args = parser.parse_args()

    with open(args.experiment_fname, 'rb') as experiment_file:
        info_dict = pickle.load(experiment_file)
    info_dict['args'].__dict__.update(args.__dict__)
    args = info_dict.pop('args')
    globals().update(info_dict)

    with open(args.reset_states_fname, 'rb') as states_file:
        reset_states = pickle.load(states_file)
        if isinstance(reset_states, dict):
            car_model_name = reset_states['car_model_name']
            reset_states = reset_states['reset_states']
        else:
            if os.path.basename(args.reset_states_fname) == 'reset_states_hard.pkl':
                car_model_name = ['kia_rio_silver', 'kia_rio_yellow', 'mitsubishi_lancer_evo']
            else:
                car_model_name = None

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
    servoing_pol = policy.TheanoServoingPolicy(predictor, alpha=1.0)

    best_theta = thetas[np.argmin(esd_costs)]
    best_w, best_lambda, best_bias = np.split(best_theta, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))
    best_bias, = best_bias
    servoing_pol.w, servoing_pol.lambda_, servoing_pol.bias = best_w, best_lambda, best_bias

    if args.num_trajs_to_save:
        output_dir = os.path.join('experiments/images100_1000',
                                  os.path.splitext(os.path.basename(args.reset_states_fname))[0],
                                  os.path.splitext(os.path.basename(args.experiment_fname))[0])
        if not os.path.exists(os.path.split(output_dir)[0]):
            os.mkdir(os.path.split(output_dir)[0])
        container = utils.container.ImageDataContainer(output_dir, 'x')
        container.reserve(['image'], (args.num_trajs_to_save, args.num_steps + 1))
        if args.save_tpv:
            container.reserve(['tpv_image'], (args.num_trajs_to_save, args.num_steps + 1))

    if args.record_file and not args.visualize:
        args.visualize = 1
    if args.visualize:
        rows, cols = 1, 3
        labels = [predictor.input_names[0], predictor.input_names[0] + ' next', predictor.input_names[0] + ' target']
        if args.visualize > 1:
            single_feature = isinstance(predictor.feature_name, str)
            rows += 1 if single_feature else len(predictor.feature_name)
            cols += 1
            if single_feature:
                assert isinstance(predictor.next_feature_name, str)
                feature_names = [predictor.feature_name]
                next_feature_names = [predictor.next_feature_names]
            else:
                assert len(predictor.feature_name) == len(predictor.next_feature_name)
                feature_names = predictor.feature_name
                next_feature_names = predictor.next_feature_name
            labels.insert(2, '')
            for feature_name, next_feature_name in zip(feature_names, next_feature_names):
                labels += [feature_name, feature_name + ' next', next_feature_name, feature_name + ' target']
        fig = plt.figure(figsize=(4 * cols, 4 * rows), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(args.experiment_fname)
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], rows * cols, rows=rows, cols=cols, labels=labels)
        plt.show(block=False)
        if args.record_file:
            FFMpegWriter = manimation.writers['ffmpeg']
            writer = FFMpegWriter(fps=1.0 / env.dt)
            writer.setup(fig, args.record_file, fig.dpi)
    else:
        image_visualizer = None

    np.random.seed(7)
    quiet = True
    test_costs = []
    for traj_iter, reset_state in enumerate(reset_states):
        print('test_rollouts. traj_iter', traj_iter)
        if args.save_tpv:
            (states, actions, rewards, next_states, target_actions), tpv_images = do_rollout(predictor, env, reset_state,
                                                                                           servoing_pol,
                                                                                           target_pol,
                                                                                           args.num_steps,
                                                                                           visualize=args.visualize,
                                                                                           image_visualizer=image_visualizer,
                                                                                           target_distance=args.target_distance,
                                                                                           quiet=quiet,
                                                                                           reset_after_target=args.reset_after_target,
                                                                                           ret_tpv=True)
        else:
            states, actions, rewards, next_states, target_actions = do_rollout(predictor, env, reset_state, servoing_pol,
                                                                               target_pol,
                                                                               args.num_steps,
                                                                               visualize=args.visualize,
                                                                               image_visualizer=image_visualizer,
                                                                               target_distance=args.target_distance,
                                                                               quiet=quiet,
                                                                               reset_after_target=args.reset_after_target)
        test_cost = np.array(rewards).dot(gamma ** np.arange(len(rewards)))
        test_costs.append(test_cost)

        if traj_iter < args.num_trajs_to_save:
            for step_iter, state in enumerate(states):
                image = state[0][0]
                image = predictor.transformers['x'].transformers[-1].deprocess(image)
                container.add_datum(traj_iter, step_iter, image=image)
            if args.save_tpv:
                for step_iter, tpv_image in enumerate(tpv_images):
                    container.add_datum(traj_iter, step_iter, tpv_image=tpv_image)

    cost_str_output = '\n'.join([str(test_cost) for test_cost in test_costs])
    print(cost_str_output)

    # cost_fname = os.path.join('experiments/costs',
    #                           os.path.splitext(os.path.basename(args.reset_states_fname))[0],
    #                           os.path.splitext(os.path.basename(args.experiment_fname))[0],
    #                           '.txt')
    # if not os.path.exists(os.path.split(cost_fname)[0]):
    #     os.mkdir(os.path.split(cost_fname)[0])
    # with open(cost_fname, 'w') as cost_file:
    #     cost_file.write(cost_str_output)

    if args.num_trajs_to_save:
        container.close()


if __name__ == '__main__':
    main()
