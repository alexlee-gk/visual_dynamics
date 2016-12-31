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
    parser.add_argument('--feature_inds', '-i', type=str, help='inds of subset of features to use')
    parser.add_argument('--use_weights', '-w', action='store_true')
    args = parser.parse_args()

    predictor = utils.from_yaml(open(args.predictor_fname))
    if args.feature_inds:
        args.feature_inds = [int(ind) for ind in args.feature_inds]
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]
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
            writer = None

        # plotting
        fig = plt.figure(figsize=(12, 6), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(predictor.name)
        gs = gridspec.GridSpec(len(error_names), 1)
        plt.show(block=False)
        rms_error_plotters = []
        for i, error_name in enumerate(error_names):
            rms_error_plotters.append(LossPlotter(fig, gs[i], ['--'], labels=['rms error'], xlabel='time', ylabel=error_name + ' error'))
    else:
        image_visualizer = None
        writer = None

    if args.use_weights:
        from q_learning import QLearning
        q_learning = QLearning(servoing_pol, gamma=0.0, l2_reg=0, learn_lambda=False, experience_replay=False,
                               max_iters=1)
        if q_learning.theta.shape[0] == 98:
            q_learning.theta = np.array([0.51216966, 1396.13257092, 1336.68187695, 707.94709098,
                                       1410.63726105, 181.57650207, 0.1504327, 1.02036714,
                                       0.01485984, 1402.54908216, 0.11320663, 0.48155122,
                                       0.27810913, 0.09874477, 754.93302429, 0.02345268,
                                       1474.8808213, 0.13729508, 0.23060447, 0.89285473,
                                       0.03346773, 1406.03245055, 768.35806878, 1415.65112944,
                                       1080.9191782, 1322.24392556, 0.16550773, 0.13667446,
                                       0.3351315, 8671.42750488, 0.07802857, 1252.89808191,
                                       1032.51407717, 1408.4153073, 1401.87802342, 1253.52889466,
                                       1410.22121848, 1162.65762727, 47.24302514, 781.29083444,
                                       0.06013999, 1409.48793941, 2.30621113, 1042.2974146,
                                       598.68220829, 0.4180876, 1290.5441637, 0.09761988,
                                       1454.35895024, 3.49549696, 373.813957, 1120.27447488,
                                       0.14212151, 1409.94511305, 1340.21427544, 1410.76159532,
                                       1373.26716785, 1406.62024629, 14.05960113, 5.59673905,
                                       685.90898625, 3531.56837255, 0.35591585, 1417.61979892,
                                       1352.61828499, 1409.74625695, 1409.03257919, 1378.58481389,
                                       1410.27373587, 1369.62115266, 1188.23337592, 1288.15159435,
                                       0.25318891, 1410.2222167, 1112.09777262, 1385.68986494,
                                       1254.38091918, 809.16333704, 1395.14212167, 0.94875743,
                                       1424.7476621, 1089.75464813, 1195.65312485, 1362.42865264,
                                       4.54125437, 1410.18195166, 1401.29744175, 1410.2602865,
                                       1404.40655678, 1410.01494277, 1089.01901981, 1132.29832696,
                                       1317.86203557, 2044.16811481, 805.37765767, 1416.39372331,
                                       -2.82847055, 1.])
        elif q_learning.theta.shape[0] == 11:
            # q_learning.theta = np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.40992658,  0.        ,  0.        , -0.52272114,  1.        ])
            q_learning.theta = np.array([0.0300786, 0.00000003, 0.00000003, 0.00000224, 0.00000014, 0.00000013, 0.00000787, 0.00000061, 0.00000057, -1.19514276, 1.])
        else:
            raise ValueError

    np.random.seed(seed=7)

    from visual_servoing_weighted import rollout
    all_errors, states, actions, next_states, rewards = rollout(predictor, env, pol, args.num_trajs, args.num_steps,
                                                                container=container,
                                                                visualize=args.visualize, image_visualizer=image_visualizer,
                                                                record_file=args.record_file, writer=writer,
                                                                target_distance=args.target_distance)

    # plotting
    if args.visualize:
        all_errors = np.array(all_errors).reshape((args.num_trajs, args.num_steps + 1, -1))
        for errors, rms_errors, rms_error_plotter in zip(all_errors.transpose([2, 0, 1]),
                                                         np.sqrt(np.mean(np.square(all_errors), axis=0)).T,
                                                         rms_error_plotters):
            rms_error_plotter.update(np.r_[[rms_errors], errors])

    if args.record_file:
        writer.finish()

    env.close()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()

    # import IPython as ipy; ipy.embed()


if __name__ == "__main__":
    main()
