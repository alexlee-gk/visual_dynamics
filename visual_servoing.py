from __future__ import division, print_function
import argparse
import numpy as np
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    parser.add_argument('--target_distance', '-d', type=int, default=1)
    args = parser.parse_args()

    predictor = utils.from_yaml(open(args.predictor_fname))
    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("visual_servoing")
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
    servoing_pol = policy.ServoingPolicy(predictor, alpha=1.0, lambda_=0.0)
    pol = policy.MixedPolicy([target_pol, servoing_pol], act_probs=[0, 1], reset_probs=[1, 0])

    error_names = env.get_error_names()
    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(['target_state'] + ['target_' + sensor_name for sensor_name in env.sensor_names], args.num_trajs)
        container.reserve(env.sensor_names + ['state'] + error_names, (args.num_trajs, args.num_steps + 1))
        container.reserve(['action'], (args.num_trajs, args.num_steps))
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

        # plotting
        fig = plt.figure(figsize=(12, 6), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(predictor.name)
        gs = gridspec.GridSpec(len(error_names), 1)
        plt.show(block=False)
        rms_error_plotters = []
        for i, error_name in enumerate(error_names):
            rms_error_plotters.append(LossPlotter(fig, gs[i], ['--'], labels=['rms error'], xlabel='time', ylabel=error_name + ' error'))

    np.random.seed(seed=7)
    all_errors = []
    errors_header_format = "{:>30}" + "{:>15}" * len(error_names)
    errors_row_format = "{:>30}" + "{:>15.2f}" * len(error_names)
    print('=' * (30 + 15 * len(error_names)))
    print(errors_header_format.format("(traj_iter, step_iter)", *error_names))
    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            state = pol.reset()
            env.reset(state)

            target_state, target_obs = env.get_state_and_observe()
            servoing_pol.set_target(target_obs)

            reset_action = random_pol.act(obs=None)
            for _ in range(args.target_distance):
                env.step(reset_action)
            if isinstance(env, envs.Pr2Env):
                import rospy
                rospy.sleep(2)

            if container:
                container.add_datum(traj_iter,
                                    **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], target_obs)))
            for step_iter in range(args.num_steps):
                state, obs = env.get_state_and_observe()
                image = obs[0]

                action = pol.act(obs)
                env.step(action)  # action is updated in-place if needed

                errors = env.get_errors(target_pol.get_target_state())
                print(errors_row_format.format(str((traj_iter, step_iter)), *errors.values()))
                all_errors.append(errors.values())

                if step_iter == (args.num_steps - 1):
                    next_state, next_obs = env.get_state_and_observe()
                    next_errors = env.get_errors(target_pol.get_target_state())
                    all_errors.append(next_errors.values())

                # container
                if container:
                    container.add_datum(traj_iter, step_iter, state=state, action=action,
                                        **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                    if step_iter == (args.num_steps-1):
                        container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                            **dict(list(next_errors.items()) + list(zip(env.sensor_names, next_obs))))

                # visualization
                if args.visualize:
                    env.render()
                    next_obs = env.observe()
                    next_image = next_obs[0]
                    target_image = target_obs[0]
                    vis_images = [image, next_image, target_image]
                    for i, vis_image in enumerate(vis_images):
                        vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                    if args.visualize == 1:
                        vis_features = vis_images
                    else:
                        feature = predictor.feature(image)
                        feature_next_pred = predictor.next_feature(image, action)
                        feature_next = predictor.feature(next_image)
                        feature_target = predictor.feature(target_image)
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

    # plotting
    all_errors = np.array(all_errors).reshape((args.num_trajs, args.num_steps + 1, -1))
    for errors, rms_errors, rms_error_plotter in zip(all_errors.transpose([2, 0, 1]), np.sqrt(np.mean(np.square(all_errors), axis=0)).T,rms_error_plotters):
        rms_error_plotter.update(np.r_[[rms_errors], errors])

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
