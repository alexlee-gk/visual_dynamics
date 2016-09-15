from __future__ import division, print_function
import argparse
import numpy as np
import cv2
import envs
import policy
import utils
from q_learning import QLearning, QLearningSlow
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer
from gui.loss_plotter import LossPlotter
import _tkinter


def rollout(predictor, env, pol, num_trajs, num_steps, action_noise=None, container=None,
            visualize=None, image_visualizer=None, record_file=None, writer=None,
            reward_type='target_action', target_distance=0):
    assert len(pol.policies) == 2
    target_pol, servoing_pol = pol.policies
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(servoing_pol, policy.ServoingPolicy)

    if target_distance:
        random_pol = policy.RandomPolicy(env.action_space, env.state_space)

    error_names = env.get_error_names()
    all_errors = []
    errors_header_format = "{:>30}" + "{:>15}" * (len(error_names) + 1)
    errors_row_format = "{:>30}" + "{:>15.4f}" * (len(error_names) + 1)
    print('=' * (30 + 15 * (len(error_names) + 1)))
    print(errors_header_format.format("(traj_iter, step_iter)", *(error_names + [reward_type or 'cost'])))
    done = False
    states = []
    actions = []
    rewards = []
    next_states = []
    if reward_type == 'target_action':
        target_actions = []
    for traj_iter in range(num_trajs):
        print('traj_iter', traj_iter)
        try:
            state = pol.reset()
            env.reset(state)

            target_obs = env.observe()
            target_image = target_obs[0]
            servoing_pol.set_image_target(target_image)

            if target_distance:
                reset_action = random_pol.act(obs=None)
                for _ in range(target_distance):
                    env.step(reset_action)

            if isinstance(env, envs.Pr2Env):
                import rospy
                rospy.sleep(1)

            if container:
                container.add_datum(traj_iter,
                                    **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], target_obs)))
            for step_iter in range(num_steps):
                state, obs = env.get_state_and_observe()
                image = obs[0]

                if reward_type == 'target_action':
                    target_action = target_pol.act(obs)
                    target_actions.append(target_action)

                action = pol.act(obs)
                # TODO
                if action_noise is not None:
                    import utils.transformations as tf
                    scale = (env.action_space.high - env.action_space.low)
                    action_noise = np.random.normal(scale=scale / 5.0)
                    axis_angle = tf.axis_angle_from_matrix(tf.random_rotation_matrix())
                    axis, angle = tf.split_axis_angle(axis_angle)
                    action_noise = np.r_[action_noise[:3], axis * action_noise[3]]
                    action += action_noise
                env.step(action)  # action is updated in-place if needed

                # errors
                errors = env.get_errors(target_pol.get_target_state())
                all_errors.append(errors.values())
                if step_iter == (num_steps - 1):
                    next_state, next_obs = env.get_state_and_observe()
                    next_errors = env.get_errors(target_pol.get_target_state())
                    all_errors.append(next_errors.values())

                # states, actions, next_states, rewards
                states.append((obs, target_obs))
                actions.append(action)
                if step_iter > 0:
                    next_states.append((obs, target_obs))
                    if reward_type == 'errors':
                        reward = np.array(errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        reward = ((predictor.preprocess(image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        reward = ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        reward = np.array(errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        reward = ((target_actions[-2] - actions[-2]) ** 2).sum() / 1000.0
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                    rewards.append(reward)
                    print(errors_row_format.format(str((traj_iter, step_iter - 1)), *(all_errors[-2] + [reward])))
                if step_iter == (num_steps - 1):
                    next_image = next_obs[0]
                    next_states.append((next_obs, target_obs))
                    if reward_type == 'errors':
                        next_reward = np.array(next_errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        next_reward = ((predictor.preprocess(next_image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        next_reward = ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        next_reward = np.array(next_errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        next_reward = ((target_actions[-1] - actions[-1]) ** 2).sum() / 1000.0
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                    rewards.append(next_reward)
                    print(errors_row_format.format(str((traj_iter, step_iter)), *(all_errors[-1] + [next_reward])))

                # container
                if container:
                    container.add_datum(traj_iter, step_iter, state=state, action=action,
                                        reward=((target_actions[-1] - action) ** 2).sum() / 1000.0,
                                        **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                    if step_iter == (num_steps-1):
                        container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                            **dict(list(next_errors.items()) + list(zip(env.sensor_names, next_obs))))

                # visualization
                if visualize:
                    env.render()
                    next_obs = env.observe()
                    next_image = next_obs[0]
                    vis_images = [image, next_image, target_image]
                    for i, vis_image in enumerate(vis_images):
                        vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                    if visualize == 1:
                        vis_features = vis_images
                    else:
                        feature = predictor.feature(image)
                        feature_next_pred = predictor.next_feature(image, action)
                        feature_next = predictor.feature(next_image)
                        feature_target = predictor.feature(target_image)
                        # put all features into a flattened list
                        vis_features = [feature, feature_next_pred, feature_next, feature_target]
                        if not isinstance(predictor.feature_name, str):
                            vis_features = [vis_features[icol][irow] for irow in range(image_visualizer.rows - 1) for icol in range(image_visualizer.cols)]
                        vis_images.insert(2, None)
                        vis_features = vis_images + vis_features
                    # deprocess features if they have 3 channels (useful for RGB images)
                    for i, vis_feature in enumerate(vis_features):
                        if vis_feature is None:
                            continue
                        if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                            vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                    try:
                        image_visualizer.update(vis_features)
                        if record_file:
                            writer.grab_frame()
                    except _tkinter.TclError:  # TODO: is this the right exception?
                        done = True
                if done:
                    break
            if done:
                break
        except KeyboardInterrupt:
            break
    print('-' * (30 + 15 * (len(error_names) + 1)))
    rms_errors = np.sqrt(np.mean(np.square(all_errors), axis=0))
    rms_reward = np.sqrt(np.mean(np.square(rewards), axis=0))
    print(errors_row_format.format("RMS", *np.r_[rms_errors, rms_reward]))
    return all_errors, states, actions, next_states, rewards


def entropy_gaussian(variance):
    return .5 * (1 + np.log(2 * np.pi)) + .5 * np.log(variance)


def compute_information_gain(predictor, env, num_trajs, num_steps, scale=0, use_features=True, learn_masks=False, visualize=None):
    policy_config = predictor.policy_config
    replace_config = {'env': env}
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
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(random_pol, policy.RandomPolicy)

    if visualize:
        fig = plt.figure(figsize=(4, 4), frameon=False, tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], 1)
        plt.show(block=False)

    images = []
    labels = []
    done = False
    for car_visible in [False, True]:
        env.car_env.car_node.setVisible(car_visible)
        for traj_iter in range(num_trajs):
            state = pol.reset()
            env.reset(state)
            for step_iter in range(num_steps):
                obs = env.observe()
                image = obs[0]
                action = pol.act(obs)
                env.step(action)  # action is updated in-place if needed

                images.append(image)
                labels.append(car_visible)

                if visualize:
                    vis_images = [predictor.preprocess(image)[0]]
                    vis_features = vis_images
                    for i, vis_feature in enumerate(vis_features):
                        if vis_feature is None:
                            continue
                        if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                            vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                    try:
                        image_visualizer.update(vis_features)
                    except _tkinter.TclError:  # TODO: is this the right exception?
                        done = True
                if done:
                    break
            if done:
                break
        if done:
            break

    images = np.array(images)
    labels = np.array(labels)
    if use_features:
        Xs = []
        ind = 0
        while ind < len(images):
            X = predictor.feature(np.array(images[ind:ind+100]))[scale]
            Xs.append(X)
            ind += 100
        X = np.concatenate(Xs)
    else:
        X = predictor.preprocess(images)[0]
    y = labels

    if learn_masks:
        std_axes = 0
    else:
        std_axes = (0, 2, 3)
    p_y1 = (y == 1).mean()
    p_y0 = 1 - p_y1
    x_std = X.std(axis=std_axes).flatten()
    x_y0_std = X[y == 0].std(axis=std_axes).flatten()
    x_y1_std = X[y == 1].std(axis=std_axes).flatten()
    information_gain = entropy_gaussian(x_std ** 2) - \
                       (p_y0 * entropy_gaussian(x_y0_std ** 2) + \
                        p_y1 * entropy_gaussian(x_y1_std ** 2))
    # assert np.allclose(information_gain, 0.5 * (np.log(x_std ** 2) - p_y0 * np.log(x_y0_std ** 2) - p_y1 * np.log(x_y1_std ** 2)))
    if learn_masks:
        information_gain = information_gain.reshape((X.shape[1:]))
        information_gain[np.isnan(information_gain)] = 0
        information_gain[np.isinf(information_gain)] = 0
    return information_gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    parser.add_argument('--target_distance', '-d', type=int, default=0)
    parser.add_argument('--feature_inds', '-i', type=int, nargs='+', help='inds of subset of features to use')
    parser.add_argument('--use_alex_weights', '-a', action='store_true')
    parser.add_argument('--use_information_gain_weights', '-g', action='store_true')

    args = parser.parse_args()

    predictor = utils.from_yaml(open(args.predictor_fname))
    # predictor.environment_config['car_color'] = 'green'
    if args.feature_inds:
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]
    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("visual_servoing_weighted")
    env = utils.from_config(predictor.environment_config)

    policy_config = predictor.policy_config
    replace_config = {'env': env}
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
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(random_pol, policy.RandomPolicy)
    servoing_pol = policy.ServoingPolicy(predictor, alpha=1.0, lambda_=1.0)
    pol = policy.MixedPolicy([target_pol, servoing_pol], act_probs=[0, 1], reset_probs=[1, 0])

    error_names = env.get_error_names()
    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(['target_state'] + ['target_' + sensor_name for sensor_name in env.sensor_names], args.num_trajs)
        container.reserve(env.sensor_names + ['state'] + error_names, (args.num_trajs, args.num_steps + 1))
        container.reserve(['action', 'reward'], (args.num_trajs, args.num_steps))
        container.add_info(environment_config=env.get_config())
        container.add_info(policy_config=pol.get_config())
    else:
        container = None

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
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], rows * cols, rows=rows, cols=cols, labels=labels)
        plt.show(block=False)
        if args.record_file:
            FFMpegWriter = manimation.writers['ffmpeg']
            writer = FFMpegWriter(fps=1.0 / env.dt)
            writer.setup(fig, args.record_file, fig.dpi)
    else:
        image_visualizer = None

    np.set_printoptions(suppress=True)
    # q_learning = QLearning(servoing_pol, gamma=0.0, l1_reg=0, learn_lambda=False, experience_replay=False, max_iters=10)
    q_learning = QLearning(servoing_pol, gamma=0.0, l2_reg=1e-6, learn_lambda=False, experience_replay=False, max_iters=1)
    if args.use_alex_weights:
        assert not args.use_information_gain_weights
        if q_learning.theta.shape[0] == 3 * 3 + 2:
            theta_init = np.zeros(3)
            theta_init[0] = 0.1
            theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
        elif q_learning.theta.shape[0] == 3 + 2:
            theta_init = np.zeros(3)
            theta_init[0] = 0.1
            theta_init = np.r_[theta_init, [0.0, q_learning.lambda_]]
        elif q_learning.theta.shape[0] == 512 * 3 + 2:
            theta_init = np.zeros(512)
            theta_init[[36, 178, 307, 490]] = 0.1
            # theta_init[[0, 1, 2, 3]] = 0.1
            # TODO: zeros for lower resolutions
            # theta_init = np.r_[theta_init, np.zeros_like(theta_init), np.zeros_like(theta_init), [0]]
            theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
        elif q_learning.theta.shape[0] == 8 * 3 + 2:
            theta_init = np.zeros(8)
            theta_init[4:8] = 3.0
            theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
        else:
            raise ValueError('alex weights are not specified')
        q_learning.theta = theta_init

    if args.use_information_gain_weights:
        assert not args.use_alex_weights
        information_gain = np.concatenate([compute_information_gain(predictor, env, args.num_trajs, args.num_steps, scale=scale) for scale in range(len(predictor.feature_name))])
        q_learning.theta = np.r_[information_gain, q_learning.theta[len(information_gain):]]

    ###
    # theta_init = np.zeros(8)
    # theta_init[4:8] = 3.0
    # theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
    # q_learning.theta = theta_init
    # np.random.set_state(exploitation_random_state)
    # all_errors, states, actions, next_states, rewards = rollout(predictor, env, pol, args.num_trajs, args.num_steps,
    #                                                             visualize=args.visualize,
    #                                                             image_visualizer=image_visualizer)
    ###

    # import IPython as ipy; ipy.embed()
    # np.random.seed(seed=7)
    # exploitation_random_state = np.random.get_state()
    #
    # information_gain = np.concatenate(
    #     [compute_information_gain(predictor, env, args.num_trajs, args.num_steps, scale=scale) for scale in
    #      range(len(predictor.feature_name))])
    # q_learning.theta = np.r_[10.0 * information_gain, q_learning.theta[len(information_gain):]]
    # np.random.set_state(exploitation_random_state)
    # all_errors, states, actions, next_states, rewards = rollout(predictor, env, pol, args.num_trajs, args.num_steps,
    #                                                            visualize=args.visualize,
    #                                                            image_visualizer=image_visualizer)

    # plotting
    fig = plt.figure(figsize=(12, 6), frameon=False, tight_layout=True)
    fig.canvas.set_window_title(predictor.name)
    gs = gridspec.GridSpec(1, 2)
    plt.show(block=False)

    rms_error_plotter = LossPlotter(fig, gs[0], labels=['rms position error', 'rms rotation error', 'rms reward', 'esd_reward'], ylabel='error')
    bellman_error_plotter = LossPlotter(fig, gs[1], labels=['bellman error'], ylabel='error')

    def do_rollout(state, action, num_steps):
        print("ROLLOUT")
        errors_header_format = "{:>30}" + "{:>15}" * len(error_names)
        errors_row_format = "{:>30}" + "{:>15.2f}" * len(error_names)
        print('=' * (30 + 15 * len(error_names)))
        print(errors_header_format.format("step_iter", *error_names))
        rewards = []

        # state = pol.reset()
        env.reset(state)

        obs_target = env.observe()
        image_target = obs_target[0]
        servoing_pol.set_image_target(image_target)

        for step_iter in range(num_steps):
            state, obs = env.get_state_and_observe()
            image = obs[0]
            if step_iter > 0:
                action = pol.act(obs)
            env.step(action)  # action is updated in-place if needed

            errors = env.get_errors(target_pol.get_target_state())
            print(errors_row_format.format(str(step_iter), *errors.values()))
            rewards.append(np.array(errors.values()).dot(np.array([1.0, 0.0])))  # TODO

            # # visualization
            # if args.visualize:
            #     env.render()
            #     obs_next = env.observe()
            #     image_next = obs_next[0]
            #     vis_images = [image, image_next, image_target]
            #     for i, vis_image in enumerate(vis_images):
            #         vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
            #     if args.visualize == 1:
            #         vis_features = vis_images
            #     else:
            #         feature = predictor.feature(image)
            #         feature_next_pred = predictor.next_feature(image, action)
            #         feature_next = predictor.feature(image_next)
            #         feature_target = predictor.feature(image_target)
            #         # put all features into a flattened list
            #         vis_features = [feature, feature_next_pred, feature_next, feature_target]
            #         if not isinstance(predictor.feature_name, str):
            #             vis_features = [vis_features[icol][irow] for irow in range(rows - 1) for icol in range(cols)]
            #         vis_images.insert(2, None)
            #         vis_features = vis_images + vis_features
            #     # deprocess features if they have 3 channels (useful for RGB images)
            #     for i, vis_feature in enumerate(vis_features):
            #         if vis_feature is None:
            #             continue
            #         if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
            #             vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
            #     image_visualizer.update(vis_features)
        return rewards

    if False:
        import IPython as ipy; ipy.embed()

        scale = 0.1
        num_trajs = 10
        num_steps = 25
        esd_rewards = []
        seed = 9
        ###### START
        if 'vgg' in predictor.name:
            slices = [[36, 178, 307, 490]]
        else:
            slices = [slice(None)]
        # for i in :
        for i in slices:
            print(i)
            np.random.seed(seed=seed)
            theta = np.zeros_like(q_learning.theta)
            theta[i] = scale
            q_learning.theta = theta

            print("USING THETA", q_learning.theta)
            print(servoing_pol.w)
            all_errors = []
            errors_header_format = "{:>30}" + "{:>15}" * len(error_names)
            errors_row_format = "{:>30}" + "{:>15.2f}" * len(error_names)
            print('=' * (30 + 15 * len(error_names)))
            print(errors_header_format.format("(traj_iter, step_iter)", *error_names))
            done = False
            states = []
            actions = []
            rewards = []
            next_states = []
            for traj_iter in range(num_trajs):
                print('traj_iter', traj_iter)
                try:
                    if traj_iter == -1:
                        state = np.array([  58.44558759,  273.34368945,   12.57534264,   -0.10993641,0.11193145,   -1.58590685,    2.5       ,    2.        ,0.        ,   -1.        ,   20.        ,   22.        ,  129])
                    else:
                        state = pol.reset()
                    env.reset(state)

                    obs_target = env.observe()
                    image_target = obs_target[0]
                    servoing_pol.set_image_target(image_target)

                    if container:
                        container.add_datum(traj_iter,
                                            **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], obs_target)))
                    for step_iter in range(num_steps):
                        if step_iter > 0:
                            prev_obs = obs
                        state, obs = env.get_state_and_observe()
                        image = obs[0]
                        # TODO
                        prev_errors = env.get_errors(target_pol.get_target_state())

                        action = pol.act(obs)
                        # NO NOISE
                        # action += np.random.normal(scale=0.1, size=action.shape)
                        # action[:3] += 10 * (np.random.random(3) - 0.5)
                        # action = env.action_space.sample()
                        env.step(action)  # action is updated in-place if needed

                        # errors
                        errors = env.get_errors(target_pol.get_target_state())
                        print(errors_row_format.format(str((traj_iter, step_iter)), *errors.values()))
                        all_errors.append(errors.values())

                        states.append((obs, obs_target))
                        actions.append(action)
                        # rewards.append(list(errors.values()))
                        # rewards.append((np.array(errors.values()) - np.array(prev_errors.values())).dot(np.array([1.0, 50.0])))
                        rewards.append(np.array(errors.values()).dot(np.array([1.0, 0.0])))  # TODO
                        if step_iter > 0:
                            next_states.append((obs, obs_target))
                            # rewards.append(((predictor.preprocess(obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(prev_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                        if step_iter == (args.num_steps-1):
                            next_state, next_obs = env.get_state_and_observe()
                            next_states.append((next_obs, obs_target))
                            # rewards.append(((predictor.preprocess(next_obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())

                        # container
                        if container:
                            if step_iter > 0:
                                container.add_datum(traj_iter, step_iter - 1, state_diff=state - prev_state)
                            container.add_datum(traj_iter, step_iter, state=state, action=action,
                                                **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                            prev_state = state
                            if step_iter == (args.num_steps-1):
                                container.add_datum(traj_iter, step_iter, state_diff=next_state - state)
                                container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                                    **dict(zip(env.sensor_names, next_obs)))

                        # visualization
                        if args.visualize:
                            env.render()
                            obs_next = env.observe()
                            image_next = obs_next[0]
                            vis_images = [image, image_next, image_target]
                            for i, vis_image in enumerate(vis_images):
                                vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                            if args.visualize == 1:
                                vis_features = vis_images
                            else:
                                feature = predictor.feature(image)
                                feature_next_pred = predictor.next_feature(image, action)
                                feature_next = predictor.feature(image_next)
                                feature_target = predictor.feature(image_target)
                                # put all features into a flattened list
                                vis_features = [feature, feature_next_pred, feature_next, feature_target]
                                if not isinstance(predictor.feature_name, str):
                                    vis_features = [vis_features[icol][irow] for irow in range(rows - 1) for icol in range(cols)]
                                vis_images.insert(2, None)
                                vis_features = vis_images + vis_features
                            # deprocess features if they have 3 channels (useful for RGB images)
                            for i, vis_feature in enumerate(vis_features):
                                if vis_feature is None:
                                    continue
                                if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                                    vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                            try:
                                image_visualizer.update(vis_features)
                                if args.record_file:
                                    writer.grab_frame()
                            except _tkinter.TclError:  # TODO: is this the right exception?
                                done = True
                        if done:
                            break
                    if done:
                        break
                except KeyboardInterrupt:
                    done = True
                    break
            if done:
                break
            esd_reward = (np.array(rewards).reshape((num_trajs, num_steps))
                          * (q_learning.gamma ** np.arange(num_steps))[None, :]).sum(axis=1).mean()
            print("Expected sum of discounted rewards", esd_reward)
            esd_rewards.append(esd_reward)
    ###### END

    rms_positions = []
    rms_rotations = []
    rms_rewards = []
    bellman_errors = []
    esd_rewards = []
    np.random.seed(seed=7)
    for i in range(50):
        print("Q_LEARNING THETA")
        print(q_learning.theta)
        print(np.sort(q_learning.theta))
        print(np.argsort(q_learning.theta))

        exploration_random_state = np.random.get_state()
        if i == 0:
            exploitation_random_state = np.random.get_state()
        else:
            np.random.set_state(exploitation_random_state)

        # rollout and get expected sum of rewards
        all_errors, states, actions, next_states, rewards = rollout(predictor, env, pol, args.num_trajs, args.num_steps,
                                                                    visualize=args.visualize, image_visualizer=image_visualizer,
                                                                    target_distance=args.target_distance)
        rms_position, rms_rotation = np.sqrt(np.mean(np.square(all_errors), axis=0))
        rms_reward = np.sqrt(np.mean(np.square(rewards), axis=0))
        rms_positions.append(rms_position)
        rms_rotations.append(rms_rotation)
        rms_rewards.append(rms_reward)
        esd_reward = (np.array(rewards).reshape((args.num_trajs, args.num_steps))
                      * (q_learning.gamma ** np.arange(args.num_steps))[None, :]).sum(axis=1).mean()
        esd_rewards.append(esd_reward)

        if False:
            for traj_iter in range(args.num_trajs):
                print('traj_iter', traj_iter)
                try:
                    state = pol.reset()
                    env.reset(state)

                    obs_target = env.observe()
                    image_target = obs_target[0]
                    servoing_pol.set_image_target(image_target)

                    if container:
                        container.add_datum(traj_iter,
                                            **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], obs_target)))
                    for step_iter in range(args.num_steps):
                        if step_iter > 0:
                            prev_obs = obs
                        state, obs = env.get_state_and_observe()
                        image = obs[0]
                        # TODO
                        prev_errors = env.get_errors(target_pol.get_target_state())

                        action = pol.act(obs)
                        # NO NOISE
                        # action += np.random.normal(scale=0.1, size=action.shape)
                        # action[:3] += 10 * (np.random.random(3) - 0.5)
                        # action = env.action_space.sample()
                        env.step(action)  # action is updated in-place if needed

                        # errors
                        errors = env.get_errors(target_pol.get_target_state())
                        print(errors_row_format.format(str((traj_iter, step_iter)), *errors.values()))
                        all_errors.append(errors.values())

                        states.append((obs, obs_target))
                        actions.append(action)
                        # rewards.append(list(errors.values()))
                        # rewards.append((np.array(errors.values()) - np.array(prev_errors.values())).dot(np.array([1.0, 50.0])))
                        # rewards.append(np.array(errors.values()).dot(np.array([1.0, 0.0])))  # TODO
                        if step_iter > 0:
                            next_states.append((obs, obs_target))
                            # rewards.append(((predictor.preprocess(obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(prev_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            rewards.append(((predictor.preprocess(obs[0])[0] - predictor.preprocess(image_target)[0]) ** 2).mean())
                        if step_iter == (args.num_steps-1):
                            next_state, next_obs = env.get_state_and_observe()
                            next_states.append((next_obs, obs_target))
                            # rewards.append(((predictor.preprocess(next_obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            rewards.append(((predictor.preprocess(next_obs[0])[0] - predictor.preprocess(image_target)[0]) ** 2).mean())

                        # container
                        if container:
                            if step_iter > 0:
                                container.add_datum(traj_iter, step_iter - 1, state_diff=state - prev_state)
                            container.add_datum(traj_iter, step_iter, state=state, action=action,
                                                **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                            prev_state = state
                            if step_iter == (args.num_steps-1):
                                container.add_datum(traj_iter, step_iter, state_diff=next_state - state)
                                container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                                    **dict(zip(env.sensor_names, next_obs)))

                        # visualization
                        if args.visualize:
                            env.render()
                            obs_next = env.observe()
                            image_next = obs_next[0]
                            vis_images = [image, image_next, image_target]
                            for i, vis_image in enumerate(vis_images):
                                vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                            if args.visualize == 1:
                                vis_features = vis_images
                            else:
                                feature = predictor.feature(image)
                                feature_next_pred = predictor.next_feature(image, action)
                                feature_next = predictor.feature(image_next)
                                feature_target = predictor.feature(image_target)
                                # put all features into a flattened list
                                vis_features = [feature, feature_next_pred, feature_next, feature_target]
                                if not isinstance(predictor.feature_name, str):
                                    vis_features = [vis_features[icol][irow] for irow in range(rows - 1) for icol in range(cols)]
                                vis_images.insert(2, None)
                                vis_features = vis_images + vis_features
                            # deprocess features if they have 3 channels (useful for RGB images)
                            for i, vis_feature in enumerate(vis_features):
                                if vis_feature is None:
                                    continue
                                if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                                    vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                            try:
                                image_visualizer.update(vis_features)
                                if args.record_file:
                                    writer.grab_frame()
                            except _tkinter.TclError:  # TODO: is this the right exception?
                                done = True
                        if done:
                            break
                    if done:
                        break
                except KeyboardInterrupt:
                    break
            esd_reward = (np.array(rewards).reshape((args.num_trajs, args.num_steps))
                          * (q_learning.gamma ** np.arange(args.num_steps))[None, :]).sum(axis=1).mean()
            esd_rewards.append(esd_reward)

        np.random.set_state(exploration_random_state)
        # rollout with noise and get (S, A, R, S') for q learning
        all_errors, states, actions, next_states, rewards = rollout(predictor, env, pol, args.num_trajs, args.num_steps,
                                                                    action_noise=True,
                                                                    visualize=args.visualize, image_visualizer=image_visualizer,
                                                                    target_distance=args.target_distance)

        if False:
            all_errors = []
            errors_header_format = "{:>30}" + "{:>15}" * len(error_names)
            errors_row_format = "{:>30}" + "{:>15.2f}" * len(error_names)
            print('=' * (30 + 15 * len(error_names)))
            print(errors_header_format.format("(traj_iter, step_iter)", *error_names))
            done = False
            states = []
            actions = []
            rewards = []
            next_states = []
            env_first_state_action = []
            for traj_iter in range(args.num_trajs):
                print('traj_iter', traj_iter)
                try:
                    state = pol.reset()
                    env.reset(state)

                    obs_target = env.observe()
                    image_target = obs_target[0]
                    servoing_pol.set_image_target(image_target)

                    if container:
                        container.add_datum(traj_iter,
                                            **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], obs_target)))
                    for step_iter in range(args.num_steps):
                        if step_iter > 0:
                            prev_obs = obs
                        state, obs = env.get_state_and_observe()
                        image = obs[0]
                        # TODO
                        prev_errors = env.get_errors(target_pol.get_target_state())

                        action = pol.act(obs)
                        import utils.transformations as tf
                        scale = (env.action_space.high - env.action_space.low)
                        action_noise = np.random.normal(scale=scale / 5.0)
                        axis_angle = tf.axis_angle_from_matrix(tf.random_rotation_matrix())
                        axis, angle = tf.split_axis_angle(axis_angle)
                        action_noise = np.r_[action_noise[:3], axis * action_noise[3]]
                        action += action_noise
                        # action += np.random.normal(scale=0.1, size=action.shape)
                        # action[:3] += 10 * (np.random.random(3) - 0.5)
                        # action = env.action_space.sample()
                        env.step(action)  # action is updated in-place if needed

                        env_first_state_action.append((state, action))

                        # errors
                        errors = env.get_errors(target_pol.get_target_state())
                        print(errors_row_format.format(str((traj_iter, step_iter)), *errors.values()))
                        all_errors.append(errors.values())

                        states.append((obs, obs_target))
                        actions.append(action)
                        # rewards.append(list(errors.values()))
                        # rewards.append((np.array(errors.values()) - np.array(prev_errors.values())).dot(np.array([1.0, 50.0])))
                        # rewards.append(np.array(errors.values()).dot(np.array([1.0, 0.0])))  # TODO
                        if step_iter > 0:
                            next_states.append((obs, obs_target))
                            # rewards.append(((predictor.preprocess(obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(prev_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            rewards.append(((predictor.preprocess(obs[0])[0] - predictor.preprocess(image_target)[0]) ** 2).mean())
                        if step_iter == (args.num_steps-1):
                            next_state, next_obs = env.get_state_and_observe()
                            next_states.append((next_obs, obs_target))
                            # rewards.append(((predictor.preprocess(next_obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            rewards.append(((predictor.preprocess(next_obs[0])[0] - predictor.preprocess(image_target)[0]) ** 2).mean())

                        # container
                        if container:
                            if step_iter > 0:
                                container.add_datum(traj_iter, step_iter - 1, state_diff=state - prev_state)
                            container.add_datum(traj_iter, step_iter, state=state, action=action,
                                                **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                            prev_state = state
                            if step_iter == (args.num_steps-1):
                                container.add_datum(traj_iter, step_iter, state_diff=next_state - state)
                                container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                                    **dict(zip(env.sensor_names, next_obs)))

                        # visualization
                        if args.visualize:
                            env.render()
                            obs_next = env.observe()
                            image_next = obs_next[0]
                            vis_images = [image, image_next, image_target]
                            for i, vis_image in enumerate(vis_images):
                                vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                            if args.visualize == 1:
                                vis_features = vis_images
                            else:
                                feature = predictor.feature(image)
                                feature_next_pred = predictor.next_feature(image, action)
                                feature_next = predictor.feature(image_next)
                                feature_target = predictor.feature(image_target)
                                # put all features into a flattened list
                                vis_features = [feature, feature_next_pred, feature_next, feature_target]
                                if not isinstance(predictor.feature_name, str):
                                    vis_features = [vis_features[icol][irow] for irow in range(rows - 1) for icol in range(cols)]
                                vis_images.insert(2, None)
                                vis_features = vis_images + vis_features
                            # deprocess features if they have 3 channels (useful for RGB images)
                            for i, vis_feature in enumerate(vis_features):
                                if vis_feature is None:
                                    continue
                                if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                                    vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                            try:
                                image_visualizer.update(vis_features)
                                if args.record_file:
                                    writer.grab_frame()
                            except _tkinter.TclError:  # TODO: is this the right exception?
                                done = True
                        if done:
                            break
                    if done:
                        break
                except KeyboardInterrupt:
                    break
            print('-' * (30 + 15 * len(error_names)))
            print(errors_row_format.format("RMS", *np.sqrt(np.mean(np.square(all_errors), axis=0))))
        bellman_error = q_learning.fit(states, actions, rewards, next_states)
        bellman_errors.append(bellman_error)

        # plotting
        rms_error_plotter.update([np.asarray(rms_positions) / 100.0, np.asarray(rms_rotations) / 10.0, rms_rewards, esd_rewards])
        bellman_error_plotter.update([bellman_errors])

    if args.record_file:
        writer.finish()

    import IPython as ipy; ipy.embed()

    env.close()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()


if __name__ == "__main__":
    main()
