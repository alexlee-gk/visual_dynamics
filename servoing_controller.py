import argparse
import cv2
import numpy as np
import controller
import target_generator
import utils


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=10, metavar='S', help='rescale image by S for visualization')

    parser.add_argument('--alpha', type=float, default=1.0, help='controller parameter')
    parser.add_argument('--lambda_', '--lambda', type=float, default=0.0, help='controller parameter')
    parser.add_argument('--dof_limit_factor', type=float, default=1.0, help='experiment parameter')
    args = parser.parse_args()

    predictor = utils.from_yaml(open(args.predictor_fname))
    env = utils.from_config(predictor.environment_config)

    # environment_config['dof_vel_limits'][0][-1] *= 0.00000001
    # environment_config['dof_vel_limits'][1][-1] *= 0.00000001
    # # environment_config['dof_vel_limits'][0][-2] *= 0.00000001
    # # environment_config['dof_vel_limits'][1][-2] *= 0.00000001

    # # TODO: feature weights
    # features = predictor.predict(predictor.feature_name, np.zeros((predictor.input_shapes[0])))
    # # #w_features = []
    # # #for i, feature in enumerate(features):
    # # #    c, h, w = feature.shape
    # # #    w_feature = np.zeros((c, w, h))
    # # #    w_feature[:, h/4:-h/4, w*3/8:-w*3/8] = np.ones((c, h/2, w/4))
    # # #    w_features.append(w_feature.flatten())
    # # #w_features = np.concatenate(w_features)
    # w_features = []
    # for i, feature in enumerate(features):
    #     c, h, w = feature.shape
    #     w_feature = np.zeros((c, w, h))
    #     if i == 2:
    #         w_feature[[13, 23, 29, 50, 63], :, :] = np.ones((h, w))
    #     w_features.append(w_feature.flatten())
    # w_features = np.concatenate(w_features)
    # import IPython as ipy; ipy.embed()

    # import IPython as ipy; ipy.embed()
    features = predictor.predict(predictor.feature_name, np.zeros((predictor.input_shapes[0])))
    w_features = []
    for i, feature in enumerate(features):
        c, h, w = feature.shape
        w_feature = np.ones((c, w, h))  # * 4**i  # TODO
        w_features.append(w_feature.flatten())
    w_features = np.concatenate(w_features)

    # target_gen = target_generator.RandomTargetGenerator(sim, args.num_trajs)
    # target_gen = target_generator.CityNodeTargetGenerator(sim, args.num_trajs)

    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(['image', 'dof_val', 'image_x0'], (args.num_trajs, args.num_steps + 1))
        container.reserve(['vel', 'image_x0_next_pred'], (args.num_trajs, args.num_steps))
        container.reserve(['image_target', 'dof_val_target', 'image_x0_target'], args.num_trajs)
        container.add_info(environment_config=sim.get_config())
        container.add_info(predictor_config=predictor.get_config())
    else:
        container = None
    """
    ### START
    if False:
        import matplotlib.pyplot as plt
        from sklearn import linear_model
        import IPython as ipy; ipy.embed()

        ctrl = controller.OgreNodeFollowerController(sim)
        num_trajs = 100
        num_steps = 10
        plt.ion()
        pos_image_targets = []
        neg_image_targets = []
        for traj_iter in range(num_trajs):
            car_dof_min, car_dof_max = [np.array([-51 - 6, 10.7, -275]), np.array([-51 + 6, 10.7, 225])]
            car_dof_values = utils.math_utils.sample_interval(car_dof_min, car_dof_max)
            car_dof_vel = [0, 0, -1]
            sim.traj_managers[0].reset(car_dof_values, car_dof_vel)

            reset_action = ctrl.step()
            sim.reset(sim.dof_values + reset_action)
            image_target = sim.observe()
            y_target = predictor.feature(image_target)
            reset_action = args.dof_limit_factor * utils.math_utils.sample_interval(*sim.dof_vel_limits)
            sim.reset(sim.dof_values + reset_action)
            for step_iter in range(num_steps):
                pos_image_target = sim.observe()
                pos_image_targets.append(pos_image_target)

                car_position = sim.ogre.getNodePosition(b'car')
                sim.ogre.setNodePosition(b'car', car_position - np.array([0, -20.0, 0]))
                neg_image_target = sim.observe()
                neg_image_targets.append(neg_image_target)
                sim.ogre.setNodePosition(b'car', car_position)

                action = ctrl.step()
                sim.apply_action(action)
                # plt.imshow(cv2.cvtColor(np.concatenate([pos_image_target, neg_image_target], axis=1), cv2.COLOR_BGR2RGB), interpolation='none')
                # plt.draw()
        pos_image_targets = np.asarray(pos_image_targets)
        neg_image_targets = np.asarray(neg_image_targets)
        pos_feature_targets = [predictor.predict(predictor.feature_name, pos_image_target) for pos_image_target in pos_image_targets]
        neg_feature_targets = [predictor.predict(predictor.feature_name, neg_image_target) for neg_image_target in neg_image_targets]

        pool = np.max
        Z_train = []
        for i, feature_targets in enumerate([pos_feature_targets, neg_feature_targets]):
            for feature_target in feature_targets:
                feature_flatten = []
                for feature in feature_target:
                    feature_flatten.extend(pool(pool(feature, axis=2), axis=1))
                Z_train.append(feature_flatten)
        Z_train = np.asarray(Z_train)
        label_train = np.r_[np.ones(len(pos_image_targets), dtype=np.int),
                            np.zeros(len(neg_image_targets), dtype=np.int)]
        regr = linear_model.LogisticRegression(penalty='l1', C=10e4)
        # data_slice = slice(3, -1)
        data_slice = slice(3+64, -1)
        regr.fit(Z_train[:, data_slice], label_train)
        w = np.zeros(Z_train.shape[1])
        w[data_slice] = np.squeeze(regr.coef_)
        w = np.maximum(w, 0)
        print("%d out of %d weights are non-zero" % ((w != 0).sum(), len(w)))
        pool_area = np.repeat([np.prod(feature.shape[1:]) for feature in features], [feature.shape[0] for feature in features])
        w_features = np.repeat(w, pool_area)
    ### END

    ### START
    if False:
        import matplotlib.pyplot as plt
        import theano
        import theano.tensor as T
        import lasagne
        import IPython as ipy; ipy.embed()

        ctrl = controller.OgreNodeFollowerController(sim)
        num_trajs = 2
        num_steps = 10
        J_y_u_y_target = []
        plt.ion()
        for traj_iter in range(num_trajs):
            car_dof_min, car_dof_max = [np.array([-51 - 6, 10.7, -275]), np.array([-51 + 6, 10.7, 225])]
            car_dof_values = utils.math_utils.sample_interval(car_dof_min, car_dof_max)
            car_dof_vel = [0, 0, -1]
            sim.traj_managers[0].reset(car_dof_values, car_dof_vel)

            reset_action = ctrl.step()
            sim.reset(sim.dof_values + reset_action)
            image_target = sim.observe()
            y_target = predictor.feature(image_target)
            reset_action = args.dof_limit_factor * utils.math_utils.sample_interval(*sim.dof_vel_limits)
            sim.reset(sim.dof_values + reset_action)
            for step_iter in range(num_steps):
                image = sim.observe()
                action = ctrl.step()
                sim.apply_action(action)
                # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation='none')
                # plt.draw()
                J, y = predictor.feature_jacobian(image, action)
                J_y_u_y_target.append((J, y, action, y_target))

        num_channels = sum(feature.shape[0] for feature in features)
        log_w_unique_var = lasagne.utils.create_param(np.zeros(num_channels), (num_channels,), name='w_unique')
        w_unique_var = T.exp(log_w_unique_var)
        num_channels = 0
        w_var = []
        for feature in features:
            C, H, W = feature.shape
            w_var.append(T.repeat(w_unique_var[num_channels:num_channels+C], H*W))
            num_channels += C
        w_var = T.concatenate(w_var)
        # w_var /= w_var.sum()

        # w_var = lasagne.utils.create_param(np.zeros(J.shape[0]), (J.shape[0],), name='w')
        # u_fn = theano.function([w], u)
        # jac = theano.gradient.jacobian(u, w)
        # jac_fn = theano.function([w], Jw)
        loss = 0
        for J, y, u, y_target in J_y_u_y_target:
            JW_var = J * w_var.dimshuffle([0, 'x'])
            u_var = T.dot(T.nlinalg.matrix_inverse(T.dot(JW_var.T, J)), T.dot(JW_var.T, y_target - y))
            loss += ((u - u_var) ** 2).mean(axis=0).sum() / 2
        l1_penalty = lasagne.regularization.apply_penalty(w_var, lasagne.regularization.l1)
        loss += l1_penalty
        learning_rate_var = theano.tensor.scalar(name='learning_rate')
        updates = lasagne.updates.sgd(loss, [log_w_unique_var], learning_rate_var)
        train_fn = theano.function([learning_rate_var], loss, updates=updates)

        predictor.plot(image, action, sim.observe(), w=w_var.eval())

    ### END
    """

    # ctrl = controller.ServoingController(predictor, alpha=args.alpha, lambda_=args.lambda_, w=w_features)

    ### BEGIN NEW ENV API
    import policy
    offset = np.array([0., -4., 3.]) * 4
    target_pol = policy.OgreCameraTargetPolicy(env, env.quad_camera_node, env.quad_node, env.car_node, offset, tightness=1.0)
    assert (w_features == 1.0).all()
    servoing_pol = policy.ServoingPolicy(predictor, alpha=args.alpha, lambda_=args.lambda_, w=w_features)
    pol = policy.MixedPolicy(target_pol, servoing_pol, act_probs=[0, 1], reset_probs=[1, 0])

    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            # dof_val_init = sim.sample_state()
            # sim.reset(dof_val_init)
            # sim.car_env.reset()
            # car_T = tf.pose_matrix(sim.car_env.car_node.getOrientation(), sim.car_env.car_node.getPosition())
            # quad_T = car_T @ tf.translation_matrix(np.array([0., -4., 3.]) * 4)
            # quad_state = tf.position_axis_angle_from_matrix(quad_T)
            # sim.reset(quad_state)
            env.reset(pol)

            image_target = env.observe()[0].copy()  # TODO: use all observations
            # TODO: observe should copy
            servoing_pol.set_image_target(image_target)
            image_target, = predictor.preprocess(image_target)

            for step_iter in range(args.num_steps):
                state = env.state
                obs = env.observe()
                image = obs[0]  # TODO: use all observations
                # action = ctrl.step(image)
                action = pol.act(image)
                env.step(action)  # action is updated in-place if needed
                if args.visualize:
                    # images = [image] + list(individual_images.values())
                    # vis_image, done = utils.visualization.visualize_images_callback(*obs, vis_scale=args.vis_scale, delay=0)
                    env.render()
                    image_next_pred = predictor.predict('x0_next_pred', image, action)
                    done, key = utils.visualization.visualize_images_callback(*predictor.preprocess(image),
                                                                              image_next_pred,
                                                                              image_target,
                                                                              image_transformer=predictor.transformers['x'].transformers[-1],
                                                                              vis_scale=args.vis_scale,
                                                                              delay=100)
                    # done, key = utils.visualize_images_callback(obs[0], obs[2], image_target, vis_scale=args.vis_scale, delay=0)
                if done:
                    break
                if key == 32:  # space
                    if container is None:
                        break
                    else:
                        print("Can't skip to next trajectory when a container is being used. Ignoring key press.")
            if done:
                break
        except KeyboardInterrupt:
            break
    ### END NEW ENV API

    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            # generate target image
            image_target, dof_val_target = target_gen.get_target()
            if container:
                image_x0_target = predictor.transformers[0].transformers[-1].deprocess(predictor.preprocess(image_target)[0])
                container.add_datum(traj_iter, image_target=image_target, dof_val_target=dof_val_target,
                                    image_x0_target=image_x0_target)
            ctrl.set_image_target(image_target)
            image_target, = predictor.preprocess(image_target)
            # image_target *= w_features[:image_target.size].reshape(image_target.shape)

            # generate initial state
            sim.reset(dof_val_target)
            ###
            # sim.ogre.setNodePosition(b'city', np.array([0, -1000, 0.0]))
            # image_target_mask = sim.observe()
            # image_target_mask = predictor.transformers[0].transformers[0].preprocess(image_target_mask)
            # image_target_mask = (image_target_mask.sum(axis=-1) > 0).astype(np.float)
            ###
            reset_action = args.dof_limit_factor * (sim.dof_vel_limits[0] + np.random.random_sample(sim.dof_vel_limits[0].shape) * (sim.dof_vel_limits[1] - sim.dof_vel_limits[0]))
            dof_val_init = sim.dof_values + reset_action
            sim.reset(dof_val_init)
            for step_iter in range(args.num_steps):
                # import IPython as ipy; ipy.embed()
                # import matplotlib.pyplot as plt
                # plt.ion()

                if False:
                    sim.ogre.setNodePosition(b'city', np.array([0, -1000, 0.0]))
                    image = sim.observe()
                    maps = predictor.maps(image)
                    image = predictor.transformers[0].transformers[0].preprocess(image)
                    image_mask = (image.sum(axis=-1) > 0).astype(np.float)
                    # image_mask *= image_target_mask

                    w_features = []
                    for i, map_ in enumerate(maps):
                        if i == 0:
                            w_features.extend([image_mask.flatten()]*3)
                            image_mask_ds = image_mask
                        else:
                            # w_features.append(np.ones_like(map_.flatten()))
                            image_mask_ds = sum([image_mask_ds[i::2, j::2] for i in [0, 1] for j in [0, 1]]) / 4.0
                            w_features.extend([image_mask_ds.flatten()]*map_.shape[0])
                    w_features = np.concatenate(w_features)
                    ctrl.w = w_features

                    # plt.imshow(image_mask)
                    # plt.draw()
                    sim.ogre.setNodePosition(b'city', np.array([0, 0, 0.0]))

                dof_val = sim.dof_values
                image = sim.observe()
                action = ctrl.step(image)
                # # import IPython as ipy; ipy.embed()
                # # import matplotlib.pyplot as plt
                #
                # w_features = []
                # maps = predictor.maps(image)
                # for map_name, map_ in zip(predictor.map_names, maps):
                #     J = predictor.jacobian(map_name, 'x0', image)
                #     J = J.reshape(map_.shape + maps[0].shape)
                #     # J = J.reshape((128, 64, 3, 1024))
                #     J = np.abs(J.mean(axis=(1, 2))).sum(axis=-3)
                #     w_features.append(J.flatten())
                #     # J = J.reshape((128, 32, 32))
                # w_features = np.concatenate(w_features)
                # # plt.imshow(utils.vis_square(J))
                # # plt.show()

                """
                # import IPython as ipy; ipy.embed()

                # x0 = predictor.input_vars[0]
                x0 = predictor.preprocess(image)[0][None, ...]
                x0 = x0.astype(theano.config.floatX)
                w0_param = theano.shared(np.zeros_like(x0, dtype=theano.config.floatX))
                w0_var = (T.tanh(w0_param) + 1.0) / 2.
                x2_var = lasagne.layers.get_output(predictor.pred_layers['x2'], inputs=x0 * w0_var, deterministic=True)
                x2 = predictor.predict('x2', image)[None, ...]
                w2 = np.zeros_like(x2, dtype=theano.config.floatX)
                w2[:, [13, 23, 29, 50, 63], :, :] += 1
                loss = ((x2 * w2 - x2_var) ** 2).mean(axis=0).sum() / 2.
                param_l1_penalty = lasagne.regularization.apply_penalty(w0_param, lasagne.regularization.l1)
                loss += 0.005 * param_l1_penalty / 2.

                updates = lasagne.updates.sgd(loss, [w0_param], 0.001)
                train_fn = theano.function([], loss, updates=updates)

                w0 = w0_var.eval()
                x1 = predictor.predict('x1', image)[None, ...]
                w1 = np.zeros_like(x1)

                predictor.plot(image, action, sim.observe(), w=np.r_[w0.flatten(), w1.flatten(), w2.flatten()])

                # image, = predictor.preprocess(image)
                # images = [image]
                # for i in range(20):
                #     image = pred
                # ictor.predict('x0_next_pred', image, action, preprocessed=True)
                #     images.append(image)
                # vis_image, done = utils.visualization.visualize_images_callback(*images,
                #                                                                 image_transformer=
                #                                                                 predictor.transformers[0].transformers[
                #                                                                     -1],
                #                                                                 vis_scale=args.vis_scale,
                #                                                                 delay=0)
                """

                action = sim.apply_action(action)
                if container:
                    image_x0 = predictor.transformers[0].transformers[-1].deprocess(predictor.preprocess(image)[0])
                    image_next_pred = predictor.predict('x0_next_pred', image, action)
                    np.clip(image_next_pred, -1.0, 1.0, out=image_next_pred)
                    image_x0_next_pred = predictor.transformers[0].transformers[-1].deprocess(image_next_pred)
                    container.add_datum(traj_iter, step_iter, image=image, dof_val=dof_val, vel=action,
                                        image_x0=image_x0, image_x0_next_pred=image_x0_next_pred)
                    if step_iter == (args.num_steps - 1):
                        image_next = sim.observe()
                        image_x0_next = predictor.transformers[0].transformers[-1].deprocess(predictor.preprocess(image)[0])
                        container.add_datum(traj_iter, step_iter + 1, image=image_next, dof_val=sim.dof_values,
                                            image_x0=image_x0_next)
                if args.visualize == 1:
                    image_next_pred = predictor.predict('x0_next_pred', image, action)
                    vis_image, done = utils.visualization.visualize_images_callback(*predictor.preprocess(image),
                                                                                    image_next_pred,
                                                                                    image_target,
                                                                                    image_transformer=predictor.transformers[0].transformers[-1],
                                                                                    vis_scale=args.vis_scale,
                                                                                    delay=100)
                elif args.visualize > 1:
                    predictor.plot(image, action, sim.observe(), w=w_features)
                if done:
                    break
            if done:
                break
        except KeyboardInterrupt:
            break
    sim.stop()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()


if __name__ == "__main__":
    main()
