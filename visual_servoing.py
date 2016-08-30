import argparse
import numpy as np
import cv2
import policy
import utils
import utils.transformations as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer
import _tkinter


"""
pixel-level
VGG
VGG with weights (weights learned before hand)
"""

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('env_fname', type=str, help='config file with environment arguments')
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=str, default=None)
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
    assert isinstance(pol.policies[-1], policy.RandomPolicy)
    assert pol.reset_probs[-1] == 0
    servoing_pol = policy.ServoingPolicy(predictor, alpha=1.0, lambda_=0.0)
    pol.policies[-1] = servoing_pol
    pol.act_probs[:] = [0] * (len(pol.act_probs) - 1) + [1]

    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(env.sensor_names + ['state'], (args.num_trajs, args.num_steps + 1))
        container.reserve('action', (args.num_trajs, args.num_steps))
        container.add_info(environment_config=env.get_config())
        container.add_info(policy_config=pol.get_config())
    else:
        container = None

    record = args.visualize and args.visualize.endswith('.mp4')
    if args.visualize:
        fig = plt.figure(figsize=(16, 12), frameon=False, tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        single_feature = isinstance(predictor.feature_name, str)
        rows = 1 if single_feature else len(predictor.feature_name)
        cols = 4
        if single_feature:
            assert isinstance(predictor.next_feature_name, str)
            feature_names = [predictor.feature_name]
            next_feature_names = [predictor.next_feature_names]
        else:
            assert len(predictor.feature_name) == len(predictor.next_feature_name)
            feature_names = predictor.feature_name
            next_feature_names = predictor.next_feature_name
        labels = []
        for feature_name, next_feature_name in zip(feature_names, next_feature_names):
            labels += [feature_name, feature_name + ' next', next_feature_name, feature_name + ' target']
        image_visualizer = GridImageVisualizer(fig, gs[0], rows * cols, rows=rows, cols=cols, labels=labels)
        plt.show(block=False)
        if record:
            FFMpegWriter = manimation.writers['ffmpeg']
            writer = FFMpegWriter(fps=1.0 / env.dt)
            writer.setup(fig, args.visualize, fig.dpi)

    np.random.seed(seed=7)
    errors = []
    error_names = ['masked image', 'position', 'rotation']
    error_header_format = "{:>15}" * (1 + len(error_names))
    error_row_format = "{:>15}" + "{:>15.2f}" * len(error_names)
    print('=' * 15 * (1 + len(error_names)))
    print(error_header_format.format("", *error_names))
    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            state = pol.reset()
            env.reset(state)

            obs_target = env.observe()
            image_target = obs_target[0]
            servoing_pol.set_image_target(image_target)

            for step_iter in range(args.num_steps):
                obs = env.observe()
                image = obs[0]
                action = pol.act(obs)
                env.step(action)  # action is updated in-place if needed

                if args.visualize:
                    env.render()
                    obs_next = env.observe()
                    image_next = obs_next[0]

                    feature = predictor.feature(image)
                    feature_next_pred = predictor.next_feature(image, action)
                    feature_next = predictor.feature(image_next)
                    feature_target = predictor.feature(image_target)

                    # put all features into a flattened list
                    vis_features = [feature, feature_next_pred, feature_next, feature_target]
                    if not isinstance(predictor.feature_name, str):
                        vis_features = [vis_features[icol][irow] for irow in range(rows) for icol in range(cols)]
                    # deprocess features if they have 3 channels (useful for RGB images)
                    for i, vis_feature in enumerate(vis_features):
                        if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                            vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                    try:
                        image_visualizer.update(vis_features)
                    except _tkinter.TclError:  # TODO: is this the right exception?
                        done = True

                    if False:
                        image_next_pred = predictor.predict('x0_next_pred', image, action)
                        # done, key = utils.visualization.visualize_images_callback(*predictor.preprocess(image),
                        #                                                           image_next_pred,
                        #                                                           image_target,
                        #                                                           image_transformer=
                        #                                                           predictor.transformers['x'].transformers[-1],
                        #                                                           vis_scale=args.vis_scale,
                        #                                                           delay=100)

                        image_masked = obs[1]
                        # image_masked_next_pred = predictor.predict('x0_next_pred', image_masked, action)
                        image_masked_target = obs_target[1]
                        image_masked_target, = predictor.preprocess(image_masked_target)
                        next_image_masked, = predictor.preprocess(env.observe()[1])


                        if 'pixel' in mode:
                            features = predictor.predict(predictor.feature_name + predictor.next_feature_name, image, action)
                            next_pred_features = features[3:]
                            features = features[:3]
                            next_features = predictor.predict(predictor.feature_name, next_image)
                            target_features = predictor.predict(predictor.feature_name, image_target, preprocessed=True)
                            images = list(zip(*[features, next_pred_features, next_features, target_features]))
                        else:
                            feature, feature_next_pred = predictor.predict(['x5', 'x5_next_pred'],
                                                                           image, action)
                            next_feature = predictor.predict('x5', next_image)
                            feature_target = predictor.predict('x5', image_target, preprocessed=True)
                            # pol.w *= np.repeat((feature_target.sum(axis=(-2, -1)) != 0.0).astype(np.float), 32 * 32)

                            try:
                                order_inds = np.argsort(q_learning.theta[:512])[::-1]
                            except NameError:
                                # order_inds = np.argsort(pol.w.reshape((512, 32 * 32, 3)).mean(axis=1).sum(axis=1))[::-1]
                                # order_inds = np.argsort(pol.w.reshape((512, 32 * 32, 3)).mean(axis=1)[:, 0])[::-1]  # red
                                order_inds = np.argsort(V[:, 0])[::-1]  # red

                            globals().update(locals())
                            images = [[predictor.preprocess(image)[0], image_next_pred, image_target],
                                      # *[[feature[ind], feature_next_pred[ind], feature_target[ind]] for ind in order_inds[:4]],
                                      [predictor.preprocess(image_masked)[0], next_image_masked, image_masked_target],
                                      [np.tensordot(V.T, feature, axes=1),
                                       np.tensordot(V_hat.T, feature_next_pred, axes=1),
                                       np.tensordot(V.T, feature_target, axes=1)]]

                        import time
                        start_time = time.time()
                        globals().update(locals())
                        images = [image for row_images in images for image in row_images]
                        images = [predictor.transformers['x'].transformers[-1].deprocess(image) for image in images]
                        images.append(visualization_camera_sensor.observe())
                        try:
                            image_visualizer.update(images)
                            if record:
                                writer.grab_frame()
                        except:
                            done = True
                        # fig, axarr = utils.draw_images_callback(images,
                        #                                         image_transformer=predictor.transformers['x'].transformers[-1],
                        #                                         num=10)
                        # print(time.time() - start_time)
                        image_masked_error = np.linalg.norm(image_masked_target - next_image_masked) ** 2
                        target_T = target_pol.target_node.getTransform()
                        target_to_offset_T = tf.translation_matrix(target_pol.offset)
                        offset_T = target_T.dot(target_to_offset_T)
                        agent_T = target_pol.agent_node.getTransform()
                        agent_to_camera_T = target_pol.camera_node.getTransform()
                        camera_T = agent_T.dot(agent_to_camera_T)
                        pos_error = np.square(offset_T[:3, 3] - camera_T[:3, 3]).sum()
                        angle_error = tf.angle_between_vectors(camera_T[:3, 2], camera_T[:3, 3] - target_T[:3, 3])
                        errors.append([image_masked_error, pos_error, angle_error])
                        print(error_row_format.format(str((traj_iter, step_iter)), *np.sqrt([image_masked_error, pos_error, angle_error])))

                        # target_T = target_pol.target_node.getTransform()
                        # target_to_offset_T = tf.translation_matrix(target_pol.offset)
                        # offset_T = target_T @ target_to_offset_T
                        # agent_T = target_pol.agent_node.getTransform()
                        # agent_to_camera_T = target_pol.camera_node.getTransform()
                        # camera_T = agent_T @ agent_to_camera_T
                        # pos_err = np.square(offset_T[:3, 3] - camera_T[:3, 3]).sum()
                        # angle = tf.angle_between_vectors(camera_T[:3, 2], camera_T[:3, 3] - target_T[:3, 3])
                        # r = 0.1 * pos_err + 1000.0 * angle ** 2
                        # print(r, pos_err, angle ** 2)

                        # feature_next = predictor.next_feature(image, action).reshape((512, 32, 32))
                        # feature_target = predictor.feature(image_target, preprocessed=True).reshape((512, 32, 32))
                        # order_inds = np.argsort(q_learning.theta)[::-1]
                        # feature_next = feature_next[order_inds, ...]
                        # feature_target = feature_target[order_inds, ...]
                        # output_pair_arr = np.array([feature_next, feature_target])

                        # plt.ion()
                        # data_min = min(feature_next.min(), feature_target.min())
                        # data_max = max(feature_next.max(), feature_target.max())
                        # plt.subplot(221)
                        # plt.imshow(utils.vis_square(feature_next, data_min=data_min, data_max=data_max))
                        # plt.subplot(222)
                        # plt.imshow(utils.vis_square(feature_target, data_min=data_min, data_max=data_max))
                        # data_min = output_pair_arr.min(axis=(0, 2, 3))[:, None, None]
                        # data_max = output_pair_arr.max(axis=(0, 2, 3))[:, None, None]
                        # plt.subplot(223)
                        # plt.imshow(utils.vis_square(feature_next, data_min=data_min, data_max=data_max))
                        # plt.subplot(224)
                        # plt.imshow(utils.vis_square(feature_target, data_min=data_min, data_max=data_max))
                        # plt.draw()


                        # feature_next = (feature_next * 255.0).astype(np.uint8)
                        # feature_target = (feature_target * 255.0).astype(np.uint8)
                        # done, key = utils.visualization.visualize_images_callback(feature_next,
                        #                                                           feature_target,
                        #                                                           window_name='features7',
                        #                                                           vis_scale=1,
                        #                                                           delay=100)
                if done:
                    break
                # if key == 32:  # space
                #     break
            if done:
                break
        except KeyboardInterrupt:
            break
    print('-' * 15 * (1 + len(error_names)))
    print(error_row_format.format("RMS", *np.sqrt(np.mean(np.array(errors), axis=0))))
    print('%.2f\t%.2f\t%.2f' % tuple([*np.sqrt(np.mean(np.array(errors), axis=0))]))
    if record:
        writer.finish()

    env.close()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()


if __name__ == "__main__":
    main()
