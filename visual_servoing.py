import argparse
import numpy as np
import cv2
import policy
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer
import _tkinter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    args = parser.parse_args()

    predictor = utils.from_yaml(open(args.predictor_fname))
    env = utils.from_config(predictor.environment_config)

    policy_config = predictor.policy_config
    replace_config = {'env': env}
    try:
        replace_config['target_env'] = env.car_env
    except AttributeError:
        pass
    pol = utils.from_config(policy_config, replace_config=replace_config)
    assert len(pol.policies) == 2
    target_pol, random_pol = pol.policies
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(random_pol, policy.RandomPolicy)
    assert pol.reset_probs[-1] == 0
    servoing_pol = policy.ServoingPolicy(predictor, alpha=1.0, lambda_=0.0)
    pol.policies[-1] = servoing_pol
    pol.act_probs[:] = [0] * (len(pol.act_probs) - 1) + [1]

    error_names = env.get_error_names()
    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(['target_' + sensor_name for sensor_name in env.sensor_names], args.num_trajs)
        container.reserve(env.sensor_names + ['state'], (args.num_trajs, args.num_steps + 1))
        container.reserve(['action', 'state_diff'] + error_names, (args.num_trajs, args.num_steps))
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

            obs_target = env.observe()
            image_target = obs_target[0]
            servoing_pol.set_image_target(image_target)

            if container:
                container.add_datum(traj_iter,
                                    **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], obs_target)))
            for step_iter in range(args.num_steps):
                state, obs = env.get_state_and_observe()
                image = obs[0]
                action = pol.act(obs)
                env.step(action)  # action is updated in-place if needed

                # errors
                errors = env.get_errors(target_pol.get_target_state())
                print(errors_row_format.format(str((traj_iter, step_iter)), *errors.values()))
                all_errors.append(errors.values())

                # container
                if container:
                    if step_iter > 0:
                        container.add_datum(traj_iter, step_iter - 1, state_diff=state - prev_state)
                    container.add_datum(traj_iter, step_iter, state=state, action=action,
                                        **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                    prev_state = state
                    if step_iter == (args.num_steps-1):
                        next_state, next_obs = env.get_state_and_observe()
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
    if args.record_file:
        writer.finish()

    env.close()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()


if __name__ == "__main__":
    main()
