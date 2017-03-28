import argparse
import time

import cv2
import joblib
import numpy as np

from visual_dynamics.utils.config import from_config, from_yaml
from visual_dynamics.utils.rl_util import discount_return, discount_returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('resume_from', type=str)
    parser.add_argument('--start_traj_iter', '-s', type=int, default=0)
    parser.add_argument('--num_trajs', '-n', type=int, default=100, metavar='N', help='number of trajectories')
    parser.add_argument('--num_steps', '-t', type=int, default=100, metavar='T', help='maximum number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--record_file', '-r', type=str)
    parser.add_argument('--reset_states_fname', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_fname', '-o', type=str)
    args = parser.parse_args()

    data = joblib.load(args.resume_from)
    algo = data['algo']
    env = data['env']
    servoing_env = env.wrapped_env._wrapped_env
    if 'image' not in servoing_env.env.sensor_names:
        env_config = servoing_env.get_config()
        env_config['env']['sensor_names'].append('image')
        servoing_env = from_config(env_config)
        env.wrapped_env._wrapped_env = servoing_env
    policy = data['policy']

    if args.reset_states_fname is None:
        reset_states = [None] * args.num_trajs
    else:
        with open(args.reset_states_fname, 'r') as reset_state_file:
            reset_state_config = from_yaml(reset_state_file)
        if reset_state_config['environment_config']['car_model_names'] != servoing_env.env.car_model_names:
            env_config = servoing_env.get_config()
            env_config['env']['car_model_names'] = reset_state_config['environment_config']['car_model_names']
            servoing_env = from_config(env_config)
            env.wrapped_env._wrapped_env = servoing_env
        reset_states = reset_state_config['reset_states']
        args.num_trajs = min(args.num_trajs, len(reset_states))

    if args.record_file:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        video_writer = cv2.VideoWriter(args.record_file, fourcc, 1.0 / servoing_env.dt, servoing_env.observation_space.spaces['image'].shape[:2][::-1])

    start_time = time.time()
    if args.verbose:
        errors_header_format = '{:>30}{:>15}'
        errors_row_format = '{:>30}{:>15.4f}'
        print(errors_header_format.format('(traj_iter, step_iter)', 'reward'))
    observations, actions, rewards = [], [], []
    frame_iter = 0
    done = False
    for traj_iter, reset_state in zip(range(args.num_trajs), reset_states):  # whichever is shorter
        if traj_iter < args.start_traj_iter:
            continue
        np.random.seed(traj_iter)
        if args.verbose:
            print('=' * 45)
        observations_, actions_, rewards_ = [], [], []

        assert not env._normalize_obs
        obs_dict = servoing_env.reset(reset_state)
        obs = env.wrapped_env._apply_transform_obs(obs_dict)
        observations_.append(obs)
        frame_iter += 1
        for step_iter in range(args.num_steps + 1):
            try:
                if args.visualize or args.record_file:
                    vis_image = obs_dict['image'].copy()
                    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                    if args.visualize:
                        cv2.imshow("image", vis_image)
                        env.render()
                        key = cv2.waitKey(10)
                        key &= 255
                        if key == 27 or key == ord('q'):
                            print("Pressed ESC or q, exiting")
                            done = True
                    if args.record_file:
                        video_writer.write(vis_image)

                if step_iter < args.num_steps:
                    action = policy.get_action(obs)[1]['mean']
                    if 'action' in env.wrapped_env._transformers:
                        scaled_action = env.wrapped_env._transformers['action'].deprocess(action)
                    else:
                        scaled_action = action
                    obs_dict, reward, episode_done, info = servoing_env.step(scaled_action)
                    obs = env.wrapped_env._apply_transform_obs(obs_dict)
                    if 'action' in env.wrapped_env._transformers:
                        action[...] = env.wrapped_env._transformers['action'].preprocess(scaled_action)
                    else:
                        action[...] = scaled_action
                    frame_iter += 1
                    observations_.append(obs)
                    actions_.append(action)
                    rewards_.append(reward)
                    if args.verbose:
                        print(errors_row_format.format(str((traj_iter, step_iter)), reward))

                if done or episode_done:
                    break
            except KeyboardInterrupt:
                break
        if args.verbose:
            print('-' * 45)
            print(errors_row_format.format('discounted return', discount_return(rewards_, algo.discount)))
            print(errors_row_format.format('return', discount_return(rewards_, 1.0)))
        observations.append(observations_)
        actions.append(actions_)
        rewards.append(rewards_)
        if done:
            break
    if args.record_file:
        video_writer.release()
    discounted_returns = discount_returns(rewards, algo.discount)
    returns = discount_returns(rewards, 1.0)
    if args.verbose:
        print('=' * 45)
        print(errors_row_format.format('mean discounted return', np.mean(discounted_returns)))
        print(errors_row_format.format('mean return', np.mean(returns)))
    else:
        print('mean discounted return: %.4f (%.4f)' % (np.mean(discounted_returns),
                                                       np.std(discounted_returns) / np.sqrt(len(discounted_returns))))
        print('mean return: %.4f (%.4f)' % (np.mean(returns),
                                            np.std(returns) / np.sqrt(len(returns))))
    env.wrapped_env.close()
    if args.record_file:
        video_writer.release()
    end_time = time.time()
    if args.verbose:
        print("average FPS: {}".format(frame_iter / (end_time - start_time)))

    if args.output_fname:
        import csv
        with open(args.output_fname, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(map(str, discounted_returns))
            writer.writerow([str(np.mean(discounted_returns)), str(np.std(discounted_returns) / np.sqrt(len(discounted_returns)))])
            writer.writerow(map(str, returns))
            writer.writerow([str(np.mean(returns)), str(np.std(returns) / np.sqrt(len(returns)))])


if __name__ == '__main__':
    main()
