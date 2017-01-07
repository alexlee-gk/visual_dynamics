from __future__ import division, print_function

import Tkinter
import time

import argparse
import matplotlib.animation as manimation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from citysim3d.envs import ServoingEnv

import envs
import policy
import utils
from gui.grid_image_visualizer import GridImageVisualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    parser.add_argument('--target_distance', '-d', type=int, default=0)
    parser.add_argument('--feature_inds', '-i', type=str, help='inds of subset of features to use')
    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor = utils.from_yaml(predictor_file)
    if args.feature_inds:
        args.feature_inds = [int(ind) for ind in args.feature_inds]
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]

    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("visual_servoing")
    # TODO: temporary to handle change in the constructor's signature
    try:
        predictor.environment_config['car_model_names'] = predictor.environment_config.pop('car_model_name')
    except KeyError:
        pass
    env = utils.from_config(predictor.environment_config)
    if not isinstance(env, ServoingEnv):
        env = ServoingEnv(env, max_time_steps=args.num_steps)

    pol = policy.ServoingPolicy(predictor)

    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(list(env.observation_space.spaces.keys()) + ['state'], (args.num_trajs, args.num_steps + 1))
        container.reserve(['action', 'reward'], (args.num_trajs, args.num_steps))
        container.add_info(environment_config=env.get_config())
        container.add_info(env_spec_config=envs.EnvSpec(env.action_space, env.observation_space).get_config())
        container.add_info(policy_config=pol.get_config())
    else:
        container = None

    if args.record_file and not args.visualize:
        args.visualize = 1
    if args.visualize:
        rows, cols = 1, 3
        labels = [predictor.input_names[0], predictor.input_names[0] + ' next', predictor.input_names[0] + ' target']
        if args.visualize > 1:
            feature_names = utils.flatten_tree(predictor.feature_name)
            next_feature_names = utils.flatten_tree(predictor.next_feature_name)
            assert len(feature_names) == len(next_feature_names)
            rows += len(feature_names)
            cols += 1
            labels.insert(2, '')
            for feature_name, next_feature_name in zip(feature_names, next_feature_names):
                labels += [feature_name, feature_name + ' next', next_feature_name, feature_name + ' target']
        fig = plt.figure(figsize=(4 * cols, 4 * rows), frameon=False, tight_layout=True)
        try:
            window_title = predictor.solvers[-1].snapshot_prefix
        except IndexError:
            window_title = predictor.name
        fig.canvas.set_window_title(window_title)
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], rows * cols, rows=rows, cols=cols, labels=labels)
        plt.show(block=False)
        if args.record_file:
            FFMpegWriter = manimation.writers['ffmpeg']
            writer = FFMpegWriter(fps=1.0 / env.dt)
            writer.setup(fig, args.record_file, fig.dpi)

    np.random.seed(seed=7)
    start_time = time.time()
    frame_iter = 0
    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            state = pol.reset()
            obs = env.reset(state)
            frame_iter += 1
            if state is None:
                state = env.get_state()
            if args.target_distance:
                raise NotImplementedError
            for step_iter in range(args.num_steps):
                if container:
                    container.add_datum(traj_iter, step_iter, state=state, **obs)

                action = pol.act(obs)
                prev_obs = obs
                obs, reward, episode_done, _ = env.step(action)  # action is updated in-place if needed
                frame_iter += 1

                if container:
                    prev_state, state = state, env.get_state()
                    container.add_datum(traj_iter, step_iter, action=action, reward=reward)
                    if step_iter == (args.num_steps - 1) or episode_done:
                        container.add_datum(traj_iter, step_iter + 1, state=state, **obs)

                if args.visualize:
                    env.render()
                    image = prev_obs['image']
                    next_image = obs['image']
                    target_image = obs['target_image']
                    vis_images = [image, next_image, target_image]
                    vis_images = list(*predictor.preprocess([np.array(vis_images)]))
                    if args.visualize == 1:
                        vis_features = vis_images
                    else:
                        feature = predictor.feature([image])
                        feature_next = predictor.feature([next_image])
                        feature_next_pred = predictor.next_feature([image, action])
                        feature_target = predictor.feature([target_image])
                        # put all features into a flattened list
                        vis_features = [feature, feature_next, feature_next_pred, feature_target]
                        vis_features = [vis_features[icol][irow] for irow in range(image_visualizer.rows - 1) for icol in range(image_visualizer.cols)]
                        vis_images.insert(2, None)
                        vis_features = vis_images + vis_features
                    try:
                        image_visualizer.update(vis_features)
                        if args.record_file:
                            writer.grab_frame()
                    except Tkinter.TclError:
                        done = True

                if done or episode_done:
                    break
            if done:
                break
        except KeyboardInterrupt:
            break
    env.close()
    if args.record_file:
        writer.finish()
    if container:
        container.close()
    end_time = time.time()
    print("average FPS: {}".format(frame_iter / (end_time - start_time)))


if __name__ == "__main__":
    main()
